interface FrameSplitterOptions {
  targetFrameSize: number
}

// Instead of passing arrays of samples, we'll pass an object
// This allows for greater versatility as samples can carry additional information
// with them, such as whether the frame is empty or not
export type Frame = {
  samples: Float32Array
  isEmpty: boolean
}

// Frame splitter is a version of the resampler which works faster and doesn't
// do any resampling. It just splits the audio into frames of a fixed size.
export class FrameSplitter {
  inputBuffer: Array<number>

  constructor(public options: FrameSplitterOptions) {
    this.inputBuffer = []
  }

  process = (audioFrame: Float32Array) => {
    const outputFrames: Array<Frame> = []

    let outputIndex = 0
    let outputFrame: Frame = {
      samples: new Float32Array(this.options.targetFrameSize),
      isEmpty: true,
    }
    for (const sample of audioFrame) {
      if (outputFrame.isEmpty && sample !== 0) outputFrame.isEmpty = false
      outputFrame.samples[outputIndex] = sample
      outputIndex++
      if (outputIndex >= this.options.targetFrameSize) {
        outputFrames.push(outputFrame)
        outputFrame = {
          samples: new Float32Array(this.options.targetFrameSize),
          isEmpty: true,
        }
        outputIndex = 0
      }
    }

    return outputFrames
  }
}
