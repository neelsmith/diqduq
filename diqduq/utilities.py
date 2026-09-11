def wrap_font(s):
    "Create an HTML string wrapping string `s` in an HTML `div` that uses the Cardo font."
    CARDO_FONT_LINK = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Cardo:wght@400;700&display=swap');"
    "</style>"
)
    return  CARDO_FONT_LINK   + "<div style=\""                + "font-family: 'Cardo', serif; direction: rtl; text-align: right; "  + "font-size: 1.5em; line-height: 1.8; unicode-bidi: plaintext;"    + f"\">{s}</div>"
        