const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const { runInNewContext } = require("node:vm");

const pluginDirectory = join(__dirname, "../voiceBalance.web");
const modelSource = readFileSync(join(pluginDirectory, "rnnoiseModel.ts"), "utf8");
const model = Buffer.from(modelSource.match(/export default "([A-Za-z0-9+/=]+)";/)[1], "base64");
const processorSource = readFileSync(join(pluginDirectory, "processor.ts"), "utf8")
    .split("String.raw`")[1].split("`;")[0].replace("${processorName}", "test");

function createProcessor(mode) {
    let Processor;
    const messages = [];
    runInNewContext(processorSource, {
        AudioWorkletProcessor: class {
            port = { postMessage: data => messages.push(data), close() {} };
        },
        registerProcessor: (_, value) => { Processor = value; },
    });
    return { processor: new Processor({ processorOptions: { mode, target: -24 } }), messages };
}

function render(processor, input) {
    const output = new Float32Array(input.length);
    for (let offset = 0; offset < input.length; offset += 128) {
        processor.process([[input.subarray(offset, offset + 128)]], [[output.subarray(offset, offset + 128)]]);
    }
    return output;
}

test("bundled RNNoise matches the pinned upstream build", () => {
    assert.equal(createHash("sha256").update(model).digest("hex"),
        "4f513a50613de74378331237886138eab52fa2650e8b1a41eb587d932d9b8850");
});

test("disabling and enabling RNNoise replaces its state after a fade", async () => {
    const { processor, messages } = createProcessor(1);
    await processor.startNoiseFilter(model);
    assert.ok(processor.state);
    render(processor, new Float32Array(4800));
    assert.equal(processor.mix, 1);

    processor.port.onmessage({ data: { mode: 0, target: -24 } });
    render(processor, new Float32Array(4800));
    assert.equal(processor.mix, 0);
    assert.equal(processor.state, 0);
    assert.equal(processor.buffer, 0);

    processor.port.onmessage({ data: { mode: 1, target: -24 } });
    await new Promise(resolve => setImmediate(resolve));
    assert.ok(processor.state);
    processor.port.onmessage({ data: "close" });
    assert.equal(processor.process([[]], [[new Float32Array(128)]]), false);
    assert.deepEqual(messages, ["ready", "primed", "closed"]);
});

test("a failing RNNoise kernel leaves the dry balance working", async () => {
    const { processor, messages } = createProcessor(1);
    await processor.startNoiseFilter(model);
    processor.rn.process = () => { throw Error("synthetic failure"); };

    const input = Float32Array.from({ length: 4800 }, (_, i) => 0.1 * Math.sin(i / 12));
    input[2400] = NaN;
    input[2401] = Infinity;
    const output = render(processor, input);
    assert.ok(output.every(Number.isFinite));
    assert.ok(output.some(sample => Math.abs(sample) > 0.03));
    assert.ok(messages.includes("noise-error"));
    assert.equal(processor.state, 0);
    processor.dispose();
});
