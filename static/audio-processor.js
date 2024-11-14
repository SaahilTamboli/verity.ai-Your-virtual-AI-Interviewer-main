class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(0);
        this.sampleRate = 48000; // Use a consistent sample rate
        this.port.onmessage = this.handleMessage.bind(this);
    }

    handleMessage(event) {
        const newData = new Float32Array(event.data);
        const newBuffer = new Float32Array(this.buffer.length + newData.length);
        newBuffer.set(this.buffer);
        newBuffer.set(newData, this.buffer.length);
        this.buffer = newBuffer;
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channelData = output[0];

        if (this.buffer.length > 0) {
            const copyLength = Math.min(channelData.length, this.buffer.length);
            channelData.set(this.buffer.subarray(0, copyLength));
            this.buffer = this.buffer.subarray(copyLength);
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);