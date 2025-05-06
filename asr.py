from funasr import AutoModel

class ASR:
    # chunk_size为流式延时配置，[0,10,5]表示上屏实时出字粒度为10*60=600ms，未来信息为5*60=300ms。每次推理输入为600ms（采样点数为16000*0.6=960），输出为对应文字，最后一个语音片段输入需要设置is_final=True来强制输出最后一个字。
    chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
    encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
    decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

    def __init__(self):
        self.model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", disable_update=True)
        self.cache = {}

    def call(self, pcm, is_final):
        res = self.model.generate(input=pcm, cache=self.cache, is_final=is_final, chunk_size=ASR.chunk_size, encoder_chunk_look_back=ASR.encoder_chunk_look_back, decoder_chunk_look_back=ASR.decoder_chunk_look_back)
        # print(res)
        if is_final:
            self.cache = {}
        return res[0]["text"]

if __name__ == "__main__":
    import soundfile

    wav_file = "./test.wav"
    speech, sample_rate = soundfile.read(wav_file)
    print(sample_rate, speech)
    chunk_stride = ASR.chunk_size[1] * 960 # 600ms

    asr = ASR()
    text = ""
    total_chunk_num = int(len((speech)-1)/chunk_stride+1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
        is_final = i == total_chunk_num - 1
        res = asr.call(speech_chunk, is_final)
        text += res
    print(text)

    text = ""
    with open("./test.wav", "rb") as f:
        f.read(44)
        speech = f.read()
    chunk_stride = chunk_stride*2 # 测试结果表明这里要*2，why?
    for i in range(0, len(speech), chunk_stride):
        res = asr.call(speech[i:i+chunk_stride], i + chunk_stride >= len(speech))
        text += res
    print(text)