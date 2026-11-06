/* SPDX-License-Identifier: GPL-3.0-or-later */

export const processorName = "vencord-voice-balance-1";

// This module runs in the audio rendering thread, outside the Vencord renderer.
export const processorSource = String.raw`
const FRAME = 480;
const LOOKAHEAD = 96;
const PCM_SCALE = 32768;
const KNEE = 0.68;
const CEILING = 0.891;
let kernel;
let kernelLoading;

function idleGate(mode) {
    return mode === 2 ? 0.04 : mode === 1 ? 0.55 : 1;
}

// Emscripten ABI of the synchronous @jitsi/rnnoise-wasm 0.2.1 build.
function loadKernel(module) {
    if (kernel) return kernel;
    let memory;
    const imports = { a: {
        a(size) {
            try {
                memory.grow(Math.ceil((size - memory.buffer.byteLength) / 65536));
                return 1;
            } catch {
                return 0;
            }
        },
        b(destination, source, size) {
            new Uint8Array(memory.buffer).copyWithin(destination, source, source + size);
        }
    } };
    const exports = new WebAssembly.Instance(module, imports).exports;
    memory = exports.c;
    exports.d();
    kernel = {
        memory, malloc: exports.e, free: exports.f, create: exports.h,
        destroy: exports.i, process: exports.j
    };
    return kernel;
}

function limit(sample) {
    const magnitude = Math.abs(sample);
    if (magnitude <= KNEE) return sample;
    const headroom = CEILING - KNEE;
    return Math.sign(sample) * (KNEE + headroom * Math.tanh((magnitude - KNEE) / headroom));
}

const PEAK_LOOKAHEAD_DECAY = Math.exp(-1 / 100);
const PEAK_RELEASE_DECAY = Math.exp(-1 / 3200);

class VoiceBalance extends AudioWorkletProcessor {
    constructor({ processorOptions: { target, mode, strength = 1 } }) {
        super();
        this.target = target;
        this.mode = mode;
        this.strength = strength;
        this.input = new Float32Array(FRAME);
        this.previousInput = new Float32Array(FRAME);
        this.output = new Float32Array(FRAME);
        this.limiterInput = new Float32Array(FRAME + LOOKAHEAD);
        this.peaks = new Float32Array(FRAME + LOOKAHEAD);
        this.offset = 0;
        this.frames = 0;
        this.gainDb = 0;
        this.transientDb = 0;
        this.noiseLevelDb = null;
        this.stationaryFrames = 0;
        this.outputGain = 1;
        this.peakEnvelope = 0;
        this.speechDb = null;
        this.pauseFrames = 400;
        this.onsetFrames = 0;
        this.hold = 0;
        this.gate = 1;
        this.mix = 0;
        this.warmup = 0;
        this.previousProbability = 0;
        this.noiseRequested = false;
        this.noiseFailed = false;
        this.closed = false;

        this.port.onmessage = ({ data }) => {
            if (this.closed) return;
            if (data === "close") {
                this.dispose();
                return;
            }
            if (data.wasm) {
                void this.startNoiseFilter(data.wasm);
                return;
            }
            this.target = data.target;
            this.mode = data.mode;
            this.strength = data.strength ?? 1;
            if (this.mode && !this.noiseRequested && !this.noiseFailed) {
                this.noiseRequested = true;
                if (kernelLoading) void this.startNoiseFilter();
                else this.port.postMessage("noise-request");
            }
        };
        this.port.postMessage("ready");
    }

    async startNoiseFilter(wasm) {
        try {
            kernelLoading ??= WebAssembly.compile(wasm).then(loadKernel);
            this.rn = await kernelLoading;
            if (this.closed) return;
            if (!this.mode) {
                this.noiseRequested = false;
                return;
            }
            this.state = this.rn.create(0);
            this.buffer = this.rn.malloc(FRAME * 4);
            if (!this.state || !this.buffer) throw new Error("RNNoise allocation failed");
            this.samples = new Float32Array(this.rn.memory.buffer, this.buffer, FRAME);
        } catch {
            if (!this.closed) this.failNoiseFilter();
        }
    }

    releaseNoiseFilter() {
        if (this.buffer) this.rn.free(this.buffer);
        if (this.state) this.rn.destroy(this.state);
        this.buffer = 0;
        this.state = 0;
        this.noiseRequested = false;
        this.warmup = 0;
        this.hold = 0;
        this.previousProbability = 0;
    }

    failNoiseFilter() {
        this.noiseFailed = true;
        this.releaseNoiseFilter();
        this.mix = 0;
        this.port.postMessage("noise-error");
    }

    dispose() {
        if (this.closed) return;
        this.closed = true;
        this.releaseNoiseFilter();
        this.port.postMessage("closed");
    }

    process(inputs, outputs) {
        if (this.closed) return false;
        const input = inputs[0][0];
        const output = outputs[0][0];
        for (let i = 0; i < output.length; i++) {
            output[i] = this.output[this.offset];
            const sample = input?.[i];
            this.input[this.offset++] = Number.isFinite(sample) ? sample : 0;
            if (this.offset === FRAME) {
                this.processFrame();
                this.offset = 0;
            }
        }
        return true;
    }

    processFrame() {
        if (++this.frames === 3) this.port.postMessage("primed");
        let filtering = Boolean(this.state && this.mode);
        let oldMix = this.mix;
        let probability = 0;
        let wetSamples;
        if (filtering || oldMix > 0) {
            try {
                this.refreshBuffer();
                for (let i = 0; i < FRAME; i++) this.samples[i] = this.input[i] * PCM_SCALE;
                probability = this.rn.process(this.state, this.buffer, this.buffer);
                if (!Number.isFinite(probability)) throw new Error("Invalid RNNoise voice probability");
                this.refreshBuffer();
                wetSamples = this.samples;
                this.warmup = Math.min(3, this.warmup + 1);
            } catch {
                this.failNoiseFilter();
                filtering = false;
                oldMix = 0;
                probability = 0;
            }
        } else if (this.state) {
            this.releaseNoiseFilter();
        }

        // RNNoise overlaps two frames. Delay the dry path by the same frame so toggling does not
        // repeat or skip speech, and warm up the model before fading it in.
        const mixTarget = filtering && this.warmup >= 3 ? 1 : 0;
        this.mix += Math.max(-0.25, Math.min(0.25, mixTarget - this.mix));
        const mixStep = (this.mix - oldMix) / FRAME;
        let mix = oldMix;
        let power = 0;
        let differencePower = 0;
        let previousSample = 0;
        for (let i = 0; i < FRAME; i++) {
            mix += mixStep;
            const dry = this.previousInput[i];
            const wet = wetSamples && wetSamples[i];
            const sample = wetSamples && Number.isFinite(wet) ? dry + (wet / PCM_SCALE - dry) * mix : dry;
            this.previousInput[i] = this.input[i];
            this.input[i] = sample;
            power += sample * sample;
            const difference = sample - previousSample;
            differencePower += difference * difference;
            previousSample = sample;
        }
        const level = 10 * Math.log10(Math.max(1e-12, power / FRAME));
        const confidence = Math.max(probability, this.previousProbability);
        this.previousProbability = probability;

        const strong = this.mode === 2;
        const threshold = strong ? 0.52 : 0.35;
        if (confidence >= threshold || (this.hold > 0 && confidence >= threshold - 0.14)) {
            this.hold = strong ? 24 : 32;
        } else {
            this.hold = Math.max(0, this.hold - 1);
        }

        // Let audible consonants open the expander even if the model has not classified them yet.
        const audibleOnset = level > -48 && confidence >= (strong ? 0.22 : 0.12);
        const open = !filtering || this.hold > 0 || audibleOnset;
        const gateTarget = open ? 1 : 1 + (idleGate(this.mode) - 1) * this.mix;
        this.gate += (gateTarget - this.gate) * (open ? 0.80 : 0.065);

        // Without a settled RNNoise model, track steady broadband sound separately. Preserve
        // the signal but prevent the leveler from treating constant hiss as a quiet speaker.
        const useVoiceDetection = filtering && this.mix >= 0.5;
        const broadband = power > 0 && differencePower > power * 0.30;
        this.noiseLevelDb ??= level;
        const noiseDelta = level - this.noiseLevelDb;
        if (!useVoiceDetection && broadband && level > -65 && Math.abs(noiseDelta) < 1.6) {
            this.stationaryFrames = Math.min(40, this.stationaryFrames + 1);
        } else {
            this.stationaryFrames = Math.max(0, this.stationaryFrames - 4);
        }
        this.noiseLevelDb += noiseDelta * 0.15;
        const background = this.stationaryFrames >= 25;
        const speech = level > -55 && !background && (!useVoiceDetection || this.hold > 0 || audibleOnset);
        if (speech && this.pauseFrames > 18) this.onsetFrames = 4;
        this.pauseFrames = speech ? 0 : this.pauseFrames + 1;
        if (speech && (!useVoiceDetection || confidence >= threshold)) {
            this.speechDb ??= level;
            // An isolated loud frame should hit the fast gain control without resetting the
            // reference used for the quieter syllables that follow it.
            const difference = level - this.speechDb;
            this.speechDb += difference > 0 ? Math.min(difference, 6) * 0.16 : difference * 0.022;
        }

        let desiredDb = this.gainDb;
        if (speech) {
            const reference = this.speechDb ?? level;
            const boost = Math.max(0, Math.min(22 * this.strength, (reference + 56) * 1.15 * Math.max(1, this.strength)));
            const correction = (this.target - reference) * Math.min(1, this.strength);
            desiredDb = Math.max(-36, Math.min(boost, correction));
            if (this.onsetFrames > 0) {
                desiredDb = Math.min(desiredDb, this.gainDb);
                this.onsetFrames--;
            }
        } else if (background || this.pauseFrames > 110) {
            desiredDb = 0;
            if (this.pauseFrames > 400) this.speechDb = null;
        }

        let speed = background ? 0.12 : 0.007;
        if (speech) {
            speed = 0.022;
            if (desiredDb < this.gainDb) speed = 0.28;
            else if (this.gainDb < 0) speed = 0.075;
        }
        if (this.strength === 0) {
            desiredDb = 0;
            speed = 0.15;
        } else {
            speed *= this.strength;
        }
        this.gainDb += (desiredDb - this.gainDb) * speed;
        // A separate fast compressor catches loud syllables without carrying their attenuation
        // into the speaker gain. The existing lookahead limiter still catches individual peaks.
        const overshoot = speech ? Math.max(0, level + this.gainDb - this.target - 4) : 0;
        const transientTarget = -Math.min(18, overshoot * 0.8 * Math.min(1, this.strength));
        const transientSpeed = transientTarget < this.transientDb ? 0.8 : 0.24;
        this.transientDb += (transientTarget - this.transientDb) * transientSpeed;
        const gain = 10 ** ((this.gainDb + this.transientDb) / 20) * this.gate;
        const gainStep = (gain - this.outputGain) / FRAME;

        // Hold back 2 ms so the limiter can see peaks on the other side of a frame boundary.
        for (let i = 0; i < FRAME; i++) {
            this.limiterInput[LOOKAHEAD + i] = this.input[i] * (this.outputGain + (i + 1) * gainStep);
        }
        let peak = 0;
        for (let i = this.limiterInput.length - 1; i >= 0; i--) {
            peak = Math.max(Math.abs(this.limiterInput[i]), peak * PEAK_LOOKAHEAD_DECAY);
            this.peaks[i] = peak;
        }
        for (let i = 0; i < FRAME; i++) {
            this.peakEnvelope = Math.max(this.peaks[i], this.peakEnvelope * PEAK_RELEASE_DECAY);
            const attenuation = this.peakEnvelope > 0.82 ? 0.82 / this.peakEnvelope : 1;
            this.output[i] = limit(this.limiterInput[i] * attenuation);
        }
        this.limiterInput.copyWithin(0, FRAME);
        this.outputGain = gain;
    }

    refreshBuffer() {
        const { buffer } = this.rn.memory;
        if (this.samples.buffer !== buffer) this.samples = new Float32Array(buffer, this.buffer, FRAME);
    }
}

registerProcessor("${processorName}", VoiceBalance);
`;
