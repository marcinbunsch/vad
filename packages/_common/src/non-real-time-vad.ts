import {
  defaultFrameProcessorOptions,
  FrameProcessor,
  FrameProcessorInterface,
  FrameProcessorOptions,
  validateOptions,
} from "./frame-processor"
import { Frame, FrameSplitter } from "./frame-splitter"
import { Message } from "./messages"
import { ModelFetcher, ONNXRuntimeAPI, Silero } from "./models"
import { Resampler } from "./resampler"

interface NonRealTimeVADSpeechData {
  audio: Float32Array | undefined
  start: number
  end: number
}

export interface NonRealTimeVADOptions extends FrameProcessorOptions {}

export const defaultNonRealTimeVADOptions: NonRealTimeVADOptions = {
  ...defaultFrameProcessorOptions,
}

export class PlatformAgnosticNonRealTimeVAD {
  frameProcessor: FrameProcessorInterface | undefined

  static async _new<T extends PlatformAgnosticNonRealTimeVAD>(
    modelFetcher: ModelFetcher,
    ort: ONNXRuntimeAPI,
    options: Partial<NonRealTimeVADOptions> = {}
  ): Promise<T> {
    const vad = new this(modelFetcher, ort, {
      ...defaultNonRealTimeVADOptions,
      ...options,
    })
    await vad.init()
    return vad as T
  }

  constructor(
    public modelFetcher: ModelFetcher,
    public ort: ONNXRuntimeAPI,
    public options: NonRealTimeVADOptions
  ) {
    validateOptions(options)
  }

  init = async () => {
    const model = await Silero.new(this.ort, this.modelFetcher)

    this.frameProcessor = new FrameProcessor(model.process, model.reset_state, {
      frameSamples: this.options.frameSamples,
      positiveSpeechThreshold: this.options.positiveSpeechThreshold,
      negativeSpeechThreshold: this.options.negativeSpeechThreshold,
      redemptionFrames: this.options.redemptionFrames,
      preSpeechPadFrames: this.options.preSpeechPadFrames,
      minSpeechFrames: this.options.minSpeechFrames,
    })
    this.frameProcessor.resume()
  }

  run = async function* (
    inputAudio: Float32Array,
    sampleRate: number
  ): AsyncGenerator<NonRealTimeVADSpeechData> {
    const resamplerOptions = {
      nativeSampleRate: sampleRate,
      targetSampleRate: 16000,
      targetFrameSize: this.options.frameSamples,
    }
    const resampler = new Resampler(resamplerOptions)
    const frames = resampler.process(inputAudio)
    let start: number, end: number
    for (const i of [...Array(frames.length)].keys()) {
      const f = frames[i]
      const { msg, audio } = await this.frameProcessor.process(f)
      switch (msg) {
        case Message.SpeechStart:
          start = (i * this.options.frameSamples) / 16
          break

        case Message.SpeechEnd:
          end = ((i + 1) * this.options.frameSamples) / 16
          // @ts-ignore
          yield { audio, start, end }
          break

        default:
          break
      }
    }
    const { msg, audio } = this.frameProcessor.endSegment()
    if (msg == Message.SpeechEnd) {
      yield {
        audio,
        // @ts-ignore
        start,
        end: (frames.length * this.options.frameSamples) / 16,
      }
    }
  }

  // This is a version of run() which runs without generators
  // it also speeds up the process by skipping resampling if the input is already at 16k
  runWithCallback = async (
    inputAudio: Float32Array,
    sampleRate: number,
    callback: (data: NonRealTimeVADSpeechData) => Promise<void>
  ) => {
    if (!this.frameProcessor) {
      throw new Error("VAD not initialized")
    }

    let frames: Frame[] = []
    // if we are already at 16k, skip resampling, go for fast frame splitting
    if (sampleRate === 16000) {
      const frameSplitter = new FrameSplitter({
        targetFrameSize: this.options.frameSamples,
      })
      frames = frameSplitter.process(inputAudio)
    } else {
      const resamplerOptions = {
        nativeSampleRate: sampleRate,
        targetSampleRate: 16000,
        targetFrameSize: this.options.frameSamples,
      }
      const resampler = new Resampler(resamplerOptions)
      const resamplerFrames = resampler.process(inputAudio)
      frames = resamplerFrames.map((f) => ({
        samples: f,
        isEmpty: !f.some((s) => s !== 0),
      }))
    }

    let start: number = 0,
      i = 0,
      end: number

    for (const frame of frames) {
      i++
      const { msg, audio } = await this.frameProcessor.process(frame)
      switch (msg) {
        case Message.SpeechStart:
          start = (i * this.options.frameSamples) / 16
          break

        case Message.SpeechEnd:
          end = ((i + 1) * this.options.frameSamples) / 16
          await callback({ audio, start, end })
          break

        default:
          break
      }
    }
    const { msg, audio } = this.frameProcessor.endSegment()
    if (msg == Message.SpeechEnd) {
      await callback({
        audio,
        start,
        end: (frames.length * this.options.frameSamples) / 16,
      })
    }
  }
}
