# rl/utils.py
from typing import List, Dict
from selenium.webdriver.common.by import By

def pick_clickable_selectors(driver, max_n=12):
    """
    Return top clickable elements as CSS selectors (best-effort).
    This is fragile but fine for controlled pages.
    """
    selected = []
    # find buttons, anchors, inputs
    elems = driver.find_elements(By.CSS_SELECTOR, "button, a, input[type=submit], input[type=button]")
    seen = set()
    for e in elems:
        try:
            css = e.get_attribute("outerHTML")[:200]
            text = (e.text or "").strip()
            visible = e.is_displayed()
            sel = None
            # try to get a stable selector (id preferred)
            eid = e.get_attribute("id")
            if eid:
                sel = {"id": eid}
            else:
                cls = e.get_attribute("class")
                if cls:
                    # fallback to css class (not unique but ok)
                    sel = {"css": f"{e.tag_name}.{cls.split()[0]}"}
            if sel is None:
                # last resort: use xpath
                sel = {"xpath": "//*"}
            if str(sel) in seen:
                continue
            seen.add(str(sel))
            selected.append({"selector": sel, "visible": visible, "text_len": len(text)})
            if len(selected) >= max_n:
                break
        except Exception:
            continue
    return selected
