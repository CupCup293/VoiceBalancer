/*
 * Vencord user plugin
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { findGroupChildrenByChildId, NavContextMenuPatchCallback } from "@api/ContextMenu";
import { definePluginSettings } from "@api/Settings";
import { Logger } from "@utils/Logger";
import definePlugin, { makeRange, OptionType } from "@utils/types";
import { Menu, UserStore } from "@webpack/common";

import { processorName, processorSource } from "./processor";
import encodedModel from "./rnnoiseModel";

interface VoiceOutput {
    id: string;
    audioContext: AudioContext;
    audioElement: HTMLAudioElement | null;
    stream: MediaStream;
    playAudio(element: HTMLAudioElement): Promise<void>;
}

interface VoiceGraph {
    userId: string;
    context: AudioContext;
    stream: MediaStream;
    track: MediaStreamTrack;
    element: HTMLAudioElement;
    source: MediaStreamAudioSourceNode;
    highpass: BiquadFilterNode;
    lowpass: BiquadFilterNode;
    meter: AnalyserNode;
    gain: GainNode;
    compressor: DynamicsCompressorNode;
    limiter: WaveShaperNode;
    destination: MediaStreamAudioDestinationNode;
    samples: Float32Array<ArrayBuffer>;
    updateRoute: () => void;
    onTrackEnded: () => void;
    correctionDb: number;
    lastSampleAt: number;
    lastSpeechAt: number;
    worklet?: AudioWorkletNode;
    workletTimeout?: number;
    failWorklet?: (error: unknown) => void;
    usingWorklet: boolean;
    routeReady: boolean;
}

const MIN_TARGET_DB = -36;
const MAX_TARGET_DB = -16;
const DEFAULT_TARGET_DB = -24;
const SPEECH_FLOOR_DB = -52;
const MAX_BOOST_DB = 18;
const PEAK_CEILING = 10 ** (-1 / 20);
const limiterCurve = new Float32Array(4097);
for (let i = 0; i < limiterCurve.length; i++) {
    const sample = 2 * i / (limiterCurve.length - 1) - 1;
    const magnitude = Math.abs(sample);
    const knee = 0.7;
    limiterCurve[i] = magnitude <= knee
        ? sample
        : Math.sign(sample) * (knee + (PEAK_CEILING - knee) * Math.tanh((magnitude - knee) / (PEAK_CEILING - knee)));
}

const settings = definePluginSettings({
    targetLevel: {
        description: "Баланс звука — целевая громкость голоса (дБ). Чем ближе к нулю, тем громче.",
        type: OptionType.SLIDER,
        markers: makeRange(MIN_TARGET_DB, MAX_TARGET_DB, 2),
        default: DEFAULT_TARGET_DB,
        stickToMarkers: true,
        onChange: updateProcessors,
    },
    noiseSuppression: {
        description: "Шумоподавление для всех собеседников без индивидуальной настройки.",
        type: OptionType.BOOLEAN,
        default: true,
        onChange: updateProcessors,
    },
    isolation: {
        description: "Сила шумоподавления. Сильная изоляция может приглушать шёпот и окончания слов.",
        type: OptionType.SELECT,
        options: [
            { label: "Шумоподавление", value: 1, default: true },
            { label: "Сильная изоляция голоса", value: 2 },
        ],
        onChange: updateProcessors,
    },
}).withPrivateSettings<{ userNoiseSuppression?: Record<string, boolean>; }>();

const overrideOptions = [
    { key: "default", label: "Как в общих настройках", value: undefined },
    { key: "on", label: "Включено", value: true },
    { key: "off", label: "Выключено", value: false },
];

const addNoiseMenu: NavContextMenuPatchCallback = (children, { user } = {}) => {
    const userId: string | undefined = user?.id;
    if (!userId || userId === UserStore.getCurrentUser()?.id) return;

    const override = settings.store.userNoiseSuppression?.[userId];
    const menu = (
        <Menu.MenuItem id="voice-balance-noise" label="Шумоподавление VoiceBalance">
            {overrideOptions.map(({ key, label, value }) => (
                <Menu.MenuRadioItem
                    key={key}
                    id={`voice-balance-noise-${key}`}
                    group="voice-balance-noise"
                    label={label}
                    checked={override === value}
                    action={() => {
                        const overrides = { ...settings.store.userNoiseSuppression };
                        if (value === undefined) delete overrides[userId];
                        else overrides[userId] = value;
                        settings.store.userNoiseSuppression = overrides;
                        for (const graph of graphs.values()) {
                            if (graph.userId === userId) updateProcessor(graph);
                        }
                    }}
                />
            ))}
        </Menu.MenuItem>
    );
    const volumeGroup = findGroupChildrenByChildId(["user-volume", "volume"], children);
    if (volumeGroup) volumeGroup.push(menu);
    else children.push(<Menu.MenuGroup>{menu}</Menu.MenuGroup>);
};

const logger = new Logger("VoiceBalance");
const graphs = new Map<VoiceOutput, VoiceGraph>();
const registeredProcessors = new WeakMap<AudioContext, Promise<void>>();
let model: Uint8Array<ArrayBuffer> | undefined;
let resamplingContext: AudioContext | undefined;
let active = false;
let timer: number | undefined;
let reportedWorkletError = false;
let reportedModelError = false;

function processorSettings(userId?: string) {
    const savedTarget = settings.store.targetLevel;
    const target = Number.isFinite(savedTarget)
        ? Math.max(MIN_TARGET_DB, Math.min(MAX_TARGET_DB, savedTarget))
        : DEFAULT_TARGET_DB;
    const override = userId ? settings.store.userNoiseSuppression?.[userId] : undefined;
    const enabled = override ?? settings.store.noiseSuppression;
    return { target, mode: enabled ? settings.store.isolation === 2 ? 2 : 1 : 0 };
}

function updateProcessors() {
    for (const graph of graphs.values()) updateProcessor(graph);
}

function reportModelError(error: unknown) {
    if (reportedModelError) return;
    reportedModelError = true;
    logger.warn("RNNoise unavailable. Voice leveling remains active.", error);
}

function updateProcessor(graph: VoiceGraph) {
    graph.worklet?.port.postMessage(processorSettings(graph.userId));
}

function syncFallbackTimer() {
    if ([...graphs.values()].some(graph => graph.routeReady && !graph.usingWorklet)) {
        timer ??= window.setInterval(adjustLevels, 50);
    } else if (timer !== undefined) {
        window.clearInterval(timer);
        timer = undefined;
    }
}

function loadProcessor(context: AudioContext) {
    let registration = registeredProcessors.get(context);
    if (!registration) {
        const url = URL.createObjectURL(new Blob([processorSource], { type: "text/javascript" }));
        registration = context.audioWorklet.addModule(url).finally(() => URL.revokeObjectURL(url));
        registeredProcessors.set(context, registration);
    }
    return registration;
}

function loadModel() {
    if (!model) {
        const decoded = atob(encodedModel);
        const bytes = new Uint8Array(decoded.length);
        for (let i = 0; i < bytes.length; i++) bytes[i] = decoded.charCodeAt(i);
        model = bytes;
    }
    return model;
}

function closeProcessor(graph: VoiceGraph) {
    window.clearTimeout(graph.workletTimeout);
    graph.workletTimeout = undefined;
    graph.failWorklet = undefined;
    const node = graph.worklet;
    graph.worklet = undefined;
    if (!node) return;
    node.onprocessorerror = null;
    // Give the processor a chance to free its per-speaker WASM state before closing the port.
    const finish = () => {
        window.clearTimeout(deadline);
        node.port.onmessage = null;
        node.port.close();
    };
    const deadline = window.setTimeout(finish, 1000);
    node.port.onmessage = ({ data }) => {
        if (data === "closed") finish();
    };
    node.port.postMessage("close");
    node.disconnect();
}

function watchWorklet(graph: VoiceGraph) {
    const fail = graph.failWorklet;
    if (!fail || graph.context.state !== "running") {
        window.clearTimeout(graph.workletTimeout);
        graph.workletTimeout = undefined;
    } else {
        graph.workletTimeout ??= window.setTimeout(() => fail(new Error("AudioWorklet did not start")), 3000);
    }
}

async function enableProcessor(output: VoiceOutput, graph: VoiceGraph) {
    const { context } = graph;
    const fallback = (error: unknown) => {
        if (graphs.get(output) !== graph) return;
        if (graph.usingWorklet) {
            graph.lowpass.disconnect();
            graph.lowpass.connect(graph.meter);
            graph.limiter.connect(graph.destination);
            graph.usingWorklet = false;
        }
        closeProcessor(graph);
        graph.routeReady = true;
        graph.updateRoute();
        syncFallbackTimer();
        if (!reportedWorkletError) {
            reportedWorkletError = true;
            logger.warn("Neural noise suppression unavailable. Using basic voice leveling.", error);
        }
    };

    graph.failWorklet = fallback;
    watchWorklet(graph);
    try {
        if (context.sampleRate !== 48000) throw new Error("RNNoise requires 48 kHz");
        if (!context.audioWorklet) throw new Error("AudioWorklet is unavailable");
        await loadProcessor(context);
        if (graphs.get(output) !== graph || graph.routeReady) return;

        const node = new AudioWorkletNode(context, processorName, {
            channelCount: 1,
            channelCountMode: "explicit",
            outputChannelCount: [1],
            processorOptions: processorSettings(graph.userId),
        });
        graph.worklet = node;
        node.onprocessorerror = event => fallback(event);
        node.port.onmessage = ({ data }) => {
            if (graph.worklet !== node) return;
            if (data === "ready") {
                graph.lowpass.disconnect();
                graph.limiter.disconnect();
                graph.usingWorklet = true;
                graph.lowpass.connect(node).connect(graph.destination);
                updateProcessor(graph);
            } else if (data === "primed") {
                graph.failWorklet = undefined;
                watchWorklet(graph);
                graph.routeReady = true;
                graph.updateRoute();
                syncFallbackTimer();
            } else if (data === "noise-request") {
                try {
                    node.port.postMessage({ wasm: loadModel() });
                } catch (error) {
                    reportModelError(error);
                }
            } else if (data === "noise-error") {
                reportModelError(data);
            }
        };
    } catch (error) {
        fallback(error);
    }
}

function processingContext(original: AudioContext) {
    if (original.sampleRate === 48000) return original;
    try {
        resamplingContext ??= new AudioContext({ sampleRate: 48000 });
        if (resamplingContext.state !== "running") {
            void resamplingContext.resume().catch(error => logger.warn("Could not resume the voice processing context", error));
        }
        return resamplingContext;
    } catch {
        return original;
    }
}

function closeUnusedResamplingContext() {
    const context = resamplingContext;
    if (!context || [...graphs.values()].some(graph => graph.context === context)) return;
    resamplingContext = undefined;
    if (context.state !== "closed") {
        void context.close().catch(error => logger.warn("Could not close the voice processing context", error));
    }
}

function adjustLevels() {
    const { target } = processorSettings();
    for (const graph of graphs.values()) {
        const { context } = graph;
        if (graph.usingWorklet || context.state !== "running" || graph.element.srcObject !== graph.destination.stream) continue;

        const now = context.currentTime;
        const elapsed = Math.min(0.1, now - graph.lastSampleAt);
        graph.lastSampleAt = now;

        graph.meter.getFloatTimeDomainData(graph.samples);
        let power = 0;
        for (const sample of graph.samples) power += sample * sample;
        const inputDb = 10 * Math.log10(power / graph.samples.length);

        let desiredDb = 0;
        let response = 0.3;
        if (inputDb >= SPEECH_FLOOR_DB) {
            graph.lastSpeechAt = now;
            // Taper the maximum boost near the noise floor.
            const boostLimit = Math.min(MAX_BOOST_DB, 1.5 * (inputDb - SPEECH_FLOOR_DB));
            desiredDb = Math.max(MIN_TARGET_DB, Math.min(boostLimit, target - inputDb));
            response = desiredDb < graph.correctionDb ? 0.035 : 0.28;
        } else if (now - graph.lastSpeechAt < 0.3) {
            continue;
        }

        graph.correctionDb += (desiredDb - graph.correctionDb) * (1 - Math.exp(-elapsed / response));
        graph.gain.gain.setTargetAtTime(10 ** (graph.correctionDb / 20), now, 0.015);
    }
}

function detach(output: VoiceOutput, restore: boolean) {
    const graph = graphs.get(output);
    if (!graph) return;

    graphs.delete(output);
    graph.context.removeEventListener("statechange", graph.updateRoute);
    output.audioContext.removeEventListener("statechange", graph.updateRoute);
    graph.track.removeEventListener("ended", graph.onTrackEnded);

    if (restore && graph.element.srcObject === graph.destination.stream) {
        graph.element.srcObject = graph.stream;
        void output.playAudio(graph.element).catch(error => logger.warn("Could not restore voice playback", error));
    }

    graph.source.disconnect();
    graph.highpass.disconnect();
    graph.lowpass.disconnect();
    closeProcessor(graph);
    graph.meter.disconnect();
    graph.gain.disconnect();
    graph.compressor.disconnect();
    graph.limiter.disconnect();
    for (const track of graph.destination.stream.getTracks()) track.stop();

    closeUnusedResamplingContext();
    syncFallbackTimer();
}

export default definePlugin({
    name: "VoiceBalance",
    description: "Версия 1.0. Выравнивает громкость собеседников. Шумоподавление можно настраивать глобально и отдельно для каждого. Для Vesktop и Vencord в браузере.",
    authors: [{ name: "Codex", id: 0n }],
    settings,
    contextMenus: { "user-context": addNoiseMenu },
    patches: [
        {
            find: "BaseWebRTCConnection.createOutput: ignoring track being torn down",
            group: true,
            replacement: [
                {
                    // A regular call has context "default". Screen shares use "stream".
                    match: /(this\.outputs\[\i\]=(\i)\),\2\.addTrack\(\i\))(?=\}destroyOutput)/,
                    replace: '$1,this.context==="default"&&$self.attach($2)',
                },
                {
                    match: /(removeTrack\(\i\)\{return this\.stream\.removeTrack\(\i\),)/,
                    replace: "$1$self.removeIfMissing(this),",
                },
                {
                    match: /destroy\(\)\{(?=this\.audioElement\?\.pause\(\))/,
                    replace: "$&$self.detach(this);",
                },
            ],
        },
    ],

    start() {
        // Migrate the old «Только баланс» profile to the separate global switch.
        if (Number(settings.store.isolation) === 0) {
            settings.store.noiseSuppression = false;
            settings.store.isolation = 1;
        }
        active = true;
    },

    stop() {
        active = false;
        for (const output of graphs.keys()) detach(output, true);
    },

    attach(output: VoiceOutput) {
        if (!active) return;

        const { audioElement, stream } = output;
        const track = stream.getAudioTracks().find(track => track.readyState === "live");
        const previous = graphs.get(output);
        if (previous && previous.track === track && previous.element === audioElement && previous.stream === stream) return;

        detach(output, true);
        if (!track || !audioElement || output.audioContext.state === "closed") return;

        try {
            const audioContext = processingContext(output.audioContext);
            // A separate source lets Discord keep its own graph for speaking indicators.
            const source = audioContext.createMediaStreamSource(new MediaStream([track]));
            const highpass = audioContext.createBiquadFilter();
            highpass.type = "highpass";
            highpass.frequency.value = 75;
            highpass.Q.value = Math.SQRT1_2;
            const lowpass = audioContext.createBiquadFilter();
            lowpass.type = "lowpass";
            lowpass.frequency.value = 12500;
            lowpass.Q.value = Math.SQRT1_2;
            const meter = audioContext.createAnalyser();
            meter.fftSize = 2048;
            const gain = audioContext.createGain();
            const compressor = audioContext.createDynamicsCompressor();
            compressor.threshold.value = -8;
            compressor.knee.value = 0;
            compressor.ratio.value = 20;
            compressor.attack.value = 0.003;
            compressor.release.value = 0.12;
            const limiter = audioContext.createWaveShaper();
            limiter.curve = limiterCurve;
            const destination = audioContext.createMediaStreamDestination();

            const updateRoute = () => {
                if (output.audioContext.state === "closed" || audioContext.state === "closed") {
                    detach(output, true);
                    return;
                }
                watchWorklet(graph);
                const playback = graph.routeReady && audioContext.state === "running" ? destination.stream : stream;
                if (audioElement.srcObject === playback) return;
                if (audioElement.srcObject !== stream && audioElement.srcObject !== destination.stream) return;

                // Keep Discord's audio element, including its mute, volume and sink settings.
                audioElement.srcObject = playback;
                void output.playAudio(audioElement).catch(error => logger.warn("Could not start voice playback", error));
            };
            const onTrackEnded = () => this.attach(output);

            const graph: VoiceGraph = {
                userId: output.id, context: audioContext, stream, track, element: audioElement, source, highpass, lowpass,
                meter, gain, compressor, limiter, destination,
                samples: new Float32Array(meter.fftSize),
                updateRoute, onTrackEnded, correctionDb: 0,
                lastSampleAt: audioContext.currentTime,
                lastSpeechAt: -Infinity,
                usingWorklet: false,
                routeReady: false,
            };
            graphs.set(output, graph);

            source.connect(highpass).connect(lowpass).connect(meter).connect(gain).connect(compressor).connect(limiter).connect(destination);
            audioContext.addEventListener("statechange", updateRoute);
            if (output.audioContext !== audioContext) output.audioContext.addEventListener("statechange", updateRoute);
            track.addEventListener("ended", onTrackEnded);
            updateRoute();
            void enableProcessor(output, graph);
        } catch (error) {
            detach(output, true);
            closeUnusedResamplingContext();
            logger.warn("Could not route incoming voice audio", error);
        }
    },

    removeIfMissing(output: VoiceOutput) {
        const graph = graphs.get(output);
        if (graph && !output.stream.getAudioTracks().includes(graph.track)) this.attach(output);
    },

    detach(output: VoiceOutput) {
        detach(output, false);
    },
});
