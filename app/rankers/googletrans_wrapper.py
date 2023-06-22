import re
import time
from googletrans import Translator
from googletrans.models import Translated


class GoogleTransWrapper:
    def __init__(self, src_lang="en", target_lang="pl") -> None:
        self.base_translator: Translator = Translator()
        self.src_lang = src_lang
        self.target_lang = target_lang

    def translate(self, text: str) -> str:
        text = self._translate(text)
        text = self._convert_24h_to_12h(text)
        return text

    def _convert_24h_to_12h(self, x):
        r_groups = re.compile(r"(?P<hour>[0-9]|0[0-9]|1[0-9]|2[0-3]):(?P<minute>[0-5][0-9])")
        pm_format_str = "{hour}:{minute} pm"

        matched_groups = [m.groupdict() for m in r_groups.finditer(x)]
        for matched_group in matched_groups:
            try:
                if int(matched_group["hour"]) <= 12:
                    continue
                new_time = pm_format_str.format(hour=int(matched_group["hour"]) - 12, minute=matched_group["minute"])
                x = x.replace(f"{matched_group['hour']}:{matched_group['minute']}", new_time)
            except ValueError:
                continue
        return x


    def _translate(self, text: str, max_attempts: int = 3) -> str:
        if max_attempts <= 0:
            print(f"Returning original text. Failed to translate {text}.")
            return text
        try:
            translated: Translated = self.base_translator.translate(text, src=self.src_lang, dest=self.target_lang)
            status_code = translated._response.status_code
            if 400 <= status_code < 500:
                raise GoogleTransWrapper400Exception(
                    f"HTTP error {status_code} - text was not translated"
                )
            return translated.text
        except:
            time.sleep(0.1)
            return self._translate(text=text, max_attempts=max_attempts-1)


class GoogleTransWrapper400Exception(Exception):
    pass
