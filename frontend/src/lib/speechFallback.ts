/** Speak the assistant reply in-browser when no PCM arrives (Edge TTS 403, etc.). */

function speakable(text: string): string {
  return text.replace(/[*_`#]+/g, " ").replace(/\s+/g, " ").trim();
}

function pickVoice(): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis?.getVoices() ?? [];
  return (
    voices.find((v) => /en-US/i.test(v.lang) && /Google|Samantha|Natural|Aria/i.test(v.name))
    ?? voices.find((v) => v.lang.startsWith("en"))
  );
}

export class SpeechFallback {
  private pending = "";
  private pcmHeard = false;

  constructor() {
    if (typeof window !== "undefined") {
      window.speechSynthesis?.getVoices();
    }
  }

  beginTurn(): void {
    this.cancelSpeak();
    this.pending = "";
    this.pcmHeard = false;
  }

  pushDelta(text: string): void {
    this.pending += text;
  }

  heardPcm(): void {
    this.pcmHeard = true;
    this.cancelSpeak();
  }

  endTurn(): void {
    if (this.pcmHeard) return;
    const text = speakable(this.pending);
    if (!text || typeof window === "undefined" || !window.speechSynthesis) return;
    const utter = new SpeechSynthesisUtterance(text);
    const voice = pickVoice();
    if (voice) utter.voice = voice;
    utter.lang = voice?.lang ?? "en-US";
    utter.rate = 1.02;
    window.speechSynthesis.speak(utter);
  }

  cancel(): void {
    this.cancelSpeak();
    this.pending = "";
    this.pcmHeard = false;
  }

  private cancelSpeak(): void {
    if (typeof window === "undefined") return;
    window.speechSynthesis?.cancel();
  }
}
