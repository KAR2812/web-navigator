from src.speech.translate import translate_text

h = "मुझे नया फोन चाहिए"
m = "मला नवीन फोन पाहिजे"

print("HI -> EN:", translate_text(h, src_lang="hi"))
print("MR -> EN:", translate_text(m, src_lang="mr"))
