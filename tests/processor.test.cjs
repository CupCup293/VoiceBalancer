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

function rmsDb(samples) {
    return 10 * Math.log10(samples.reduce((power, sample) => power + sample * sample, 0) / samples.length);
}

test("the limiter smoothly anticipates a peak at a frame boundary", () => {
    const { processor } = createProcessor(0);
    const input = new Float32Array(144000).fill(0.03);
    input[96000] = 1;
    const output = render(processor, input);

    const peak = output.findIndex(sample => sample > 0.7);
    assert.ok(peak > 96000);
    assert.ok(output[peak] < 0.83);
    assert.ok(Math.abs(output[peak - 1] - output[peak + 1]) < 0.004);
    assert.ok(output[peak - 50] > output[peak - 1]);
});

test("quiet speech recovers promptly after a short loud burst", () => {
    const { processor } = createProcessor(0);
    const sampleRate = 48000;
    const burst = 2 * sampleRate;
    const burstEnd = burst + 1440;
    const input = Float32Array.from({ length: 3 * sampleRate }, (_, i) => {
        const amplitude = i >= burst && i < burstEnd ? 0.9 : 0.03;
        return amplitude * Math.sin(i * Math.PI / 100);
    });
    const output = render(processor, input);
    const before = rmsDb(output.subarray(burst - 0.2 * sampleRate, burst - 0.1 * sampleRate));
    const after = rmsDb(output.subarray(burstEnd + 0.2 * sampleRate, burstEnd + 0.3 * sampleRate));
    assert.ok(before - after < 4, `Gain recovered to ${after.toFixed(1)} dB from ${before.toFixed(1)} dB`);
});

test("balance strength and target can change during playback", () => {
    const { processor } = createProcessor(0);
    const speech = Float32Array.from({ length: 144000 }, (_, i) => 0.014 * Math.sin(i * Math.PI / 100));
    const measure = (strength, target = -19) => {
        processor.port.onmessage({ data: { mode: 0, target, strength } });
        return rmsDb(render(processor, speech).subarray(120000));
    };

    const soft = measure(0.55);
    const normal = measure(1);
    const strong = measure(1.35);
    assert.ok(normal > soft + 4);
    assert.ok(strong > normal + 2);
    assert.ok(measure(1.35, -25) < strong - 3);
    assert.ok(Math.abs(measure(0) - rmsDb(speech)) < 0.1);
});

test("steady broadband noise stays at its input level and quiet voice still gains level", () => {
    const { processor } = createProcessor(0);
    let seed = 419351;
    const noise = Float32Array.from({ length: 96000 }, () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return (seed / 2 ** 32 - 0.5) * 0.045;
    });
    const background = render(processor, noise).subarray(72000);
    assert.ok(Math.abs(rmsDb(background) - rmsDb(noise)) < 0.5);

    const voice = Float32Array.from({ length: 96000 }, (_, i) => 0.017 * Math.sin(i * Math.PI / 100));
    const balanced = render(processor, voice).subarray(72000);
    assert.ok(rmsDb(balanced) > rmsDb(voice) + 6);
});
