export enum Message {
  AudioFrame = "AUDIO_FRAME",
  SpeechStart = "SPEECH_START",
  SpeechSegment = "SPEECH_SEGMENT", // speech segment is when we hit the max frames but still have speech
  VADMisfire = "VAD_MISFIRE",
  SpeechEnd = "SPEECH_END",
}
