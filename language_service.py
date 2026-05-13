"""
language_service.py — Multilingual input/output translation layer
=================================================================
Wraps the chatbot pipeline with Arabic / Franco-Arabic support.

Detection (no LLM — free):
  Tier 1 — Arabic Unicode range check (U+0600–U+06FF)
  Tier 2 — Franco-Arabic word list with a contextual rule for "mesh"/"msh"

  "mesh"/"msh" contextual rule:
    Rule 1 — mesh + any other Franco word            → franco-arabic
    Rule 2 — mesh + CS networking term + no Franco   → english
    Rule 3 — mesh + no CS term + no other Franco     → franco-arabic

Translation direction:
  Input   : student message  →  English  (for pipeline)
  Output  : English answer   →  Arabic   (for student)
  History : batch non-English messages → English (one LLM call max)

Language labels stored per message in Supabase:
  user messages      : "english" | "arabic" | "franco-arabic"
  assistant messages : "english" | "arabic"  (franco output is never generated)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

from debug_box import box as _box, is_verbose as _is_verbose

logger = logging.getLogger(__name__)


def _trunc(s: str, n: int = 65) -> str:
    """Truncate a string for display inside debug boxes."""
    return s[:n] + "…" if len(s) > n else s


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Franco-Arabic word list
# Only words that are unmistakably Franco and cannot appear in English
# academic advisor queries about courses, prerequisites, and planning.
#
# Excluded intentionally:
#   "bas"  — conflicts with BAS course code prefix (Basic & Applied Sciences)
#   "mesh" — conflicts with "mesh network" CS term; handled via contextual rule
#   "msh"  — abbreviated form of "mesh"; same contextual rule applies
# ─────────────────────────────────────────────────────────────────────────────
_FRANCO_WORDS = {
    # Pronouns
    "ana", "enta", "enti", "howa", "heya", "ehna", "ento", "homa",
    # Negation / discourse
    "wala", "keda", "kida", "lazem", "el",
    # Common Franco words with embedded Franco numbers
    "3ayez", "3ayz", "3aiza", "3arif", "3arfa", "3ala", "3la",
    "7aga", "7agat", "7ases", "7atta",
    "2ana", "2asl", "2awi",
    "mafish", "a5od", "bta3", "bta3et",
    # Common short Franco words (safe in academic CS context)
    "de", "da", "dol", "eli", "elli",
    "momken", "leh", "leih", "fein", "emta", "ezay", "eih",
    "yalla", "khalas", "xalas", "tamam",
    "3nd", "3ndi", "3ndo", "w",
    "aslan", "ya3ni", "ya3ny", "aywa", "la2", "aiwa",
    "mn", "fe", "fi",
}

# mesh / msh require a context-aware check — kept separate
_MESH_WORDS = {"mesh", "msh"}

# English CS terms that legitimately appear next to "mesh" / "msh"
_MESH_CS_TERMS = {
    "network", "networks", "networking",
    "topology", "topologies",
    "wifi", "wireless",
    "routing", "router", "routers",
    "node", "nodes",
    "protocol", "protocols",
    "architecture", "architectures",
    "infrastructure",
    "connection", "connections",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 + Tier 2 detection (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_with_reason(text: str) -> Tuple[str, str]:
    """
    Internal detection returning (lang, reason) for debug output.
    lang   : 'english' | 'arabic' | 'franco-arabic'
    reason : human-readable string describing which signal triggered the result
    """
    # Tier 1: any Arabic Unicode character (U+0600–U+06FF)
    if any('؀' <= c <= 'ۿ' for c in text):
        return "arabic", "Tier 1 — Arabic Unicode detected"

    words = set(re.findall(r"[a-zA-Z0-9']+", text.lower()))

    # Tier 2a: core Franco word list
    matched = words & _FRANCO_WORDS
    if matched:
        sample = ", ".join(sorted(matched)[:4])
        return "franco-arabic", f"Tier 2 — word match: {sample}"

    # Tier 2b: mesh / msh contextual rule
    if words & _MESH_WORDS:
        found_mesh = ", ".join(words & _MESH_WORDS)
        cs_hit = words & _MESH_CS_TERMS
        if cs_hit:
            return "english", f"Tier 2 — '{found_mesh}' + CS term ({', '.join(cs_hit)}) → English"
        return "franco-arabic", f"Tier 2 — '{found_mesh}' alone, no CS term → Franco"

    return "english", "Tier 1+2 — no Arabic/Franco signal"


def detect_language(text: str) -> str:
    """
    Returns 'english', 'arabic', or 'franco-arabic'. Never calls an LLM.
    """
    return _detect_with_reason(text)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level translate LLM caller (reuses llm_client key-fallback pool)
# ─────────────────────────────────────────────────────────────────────────────

def _call_translate(messages: list, max_tokens: int = 800) -> str:
    from llm_client import llm_call, GROQ_MODEL_TRANSLATE
    return llm_call(messages, temperature=0, max_tokens=max_tokens, model=GROQ_MODEL_TRANSLATE)


# ─────────────────────────────────────────────────────────────────────────────
# Input translation
# ─────────────────────────────────────────────────────────────────────────────

# ─── Constants (module-level, not rebuilt on every call) ──────────────────────

_FRANCO_DECODING_KEY = """\
=== FRANCO-ARABIC DECODING KEY ===

NUMBER → ARABIC LETTER SUBSTITUTIONS:
  2 → ء  (hamza / glottal stop)        e.g. "2olta" = قلتها
  3 → ع  (ain, voiced pharyngeal)      e.g. "3arif" = عارف
  4 → ش  (sh sound)                    e.g. "34an"  = عشان
  5 → خ  (kh, like Scottish "loch")    e.g. "5alas" = خلاص
  6 → ط  (emphatic t)                  e.g. "6ab3an" = طبعاً
  7 → ح  (strong h, like Arabic ح)     e.g. "7aga"  = حاجة
  8 → غ  (gh, like French 'r')         e.g. "8alat" = غلط
  9 → ص  (emphatic s)                  e.g. "9a7"   = صح

LETTER CLUSTERS:
  sh → ش    kh → خ    gh → غ    dh → ذ    th → ث
  ou/oo → و (long u)    ee/ii → ي (long i)
  el/il/al → ال (the)   -t/-et suffix = past tense feminine / noun

NEGATION (wrap-around):
  ma___sh / m___sh → مـ...ش   e.g. "ma3rafsh" = I don't know
  msh / mesh / mish → مش (standalone "not")

PRONOUNS & POSSESSIVES:
  ana=I      enta/enti=you(m/f)   e7na/e7na=we
  howa=he    heya=she             homma=they
  -i/-y = my    -k = your    -h/-o = his/her/its    -na = our    -kom = your(pl)

TENSE MARKERS:
  ha- / h- prefix  → future        e.g. "haro7" = I will go
  bi- prefix       → present cont. e.g. "biyeshtghal" = he is working
  past             → no prefix, just conjugated verb

QUESTION WORDS:
  eh=what   meen=who    feen=where  emta=when
  izzay=how  leeh=why   2ad eh=how much   kام=how many

KEY VOCABULARY (common in academic/student context):
  lazem        = must / need to         mesh fahem   = don't understand
  3ayez/3awza  = want (m/f)             khalas/5alas = done / finished / that's it
  2adr/a2dar   = can / able to          momken       = maybe / possible / can
  3shan/34an   = because / so that      bas          = but / only / enough
  keda         = like this / this way   ya3ni        = I mean / meaning / like
  ta3ala/tala  = come                   mesh/msh     = not
  feeh         = there is / there are   mafeesh      = there isn't / none
  delwa2ti     = now / right now        ba3den       = later / after
  aho/ahe      = there it is / here     yalla        = let's go / come on
  3malt        = did / made             2olt         = said
"""

_FRANCO_FEW_SHOTS = """\
# GPA / Student Info
"2ana GPA 2ad eh"                                        → "What is my GPA?"
"eh elly 5lst-uh lel7d da"                               → "What have I completed so far?"
"2ad eh credits 5lst-ha"                                 → "How many credits have I completed?"

# Course Info
"el machine learning byt3alem feeh eh"                  → "What is machine learning about?"
"2ad eh credits el deep learning"                        → "How many credits is deep learning?"

# Prerequisites / Dependencies
"el ML byt2fl eh"                                        → "What does ML close/unlock?"
"el prerequisites bta3t el OS eh"                        → "What are the prerequisites for OS?"
"2ana 3dit AIM301 2dar a5od eh dilwa2ti"                → "I passed AIM301, what can I take now?"
"el SE byt2fl 2ad eh mwad"                               → "What courses does SE unlock?"

# Course Timing
"el machine learning byt3alem emta"                      → "When is machine learning taught?"
"el OS fi anhy semester"                                 → "Which semester is OS offered?"

# Eligibility
"a2dar a5od el ML"                                       → "Can I take ML?"
"ana mstw7ash el OS"                                     → "Am I eligible for OS?"

# Courses by Term
"eh elly byadros-uh fi sena 2 semester 1"               → "What's studied in year 2 semester 1?"
"mwad el sena el talta eh"                               → "What courses are in year 3?"

# Electives
"eh el electives fi el AI track"                         → "What electives are in the AI track?"
"emta a2dar a5od electives w 2ad eh slots"              → "When can I take electives and how many slots are there?"

# Filter / Credit breakdown
"wareni el mwad el 3 credits fi AIM"                    → "Show me the 3-credit courses in AIM"
"el core courses fi SAD eh"                              → "What are the core courses in SAD?"
"el 136 credits mwaz3a izzay"                           → "How are the 136 credits distributed?"
"2ad eh credits el humanities"                          → "How many humanities credits are there?"

# Graduation / Program Info
"3ayez 2ad eh credits 34an atkharg"                     → "How many credits do I need to graduate?"
"el minimum GPA lel graduation eh"                      → "What is the minimum GPA for graduation?"
"2oly 3an el AIM program"                               → "Tell me about the AIM program"
"eh el far2 been AIM w SAD"                             → "What is the difference between AIM and SAD?"

# Curriculum
"wareni el full curriculum bta3 el AIM"                 → "Show me the full curriculum for AIM"
"eh el mwad el shared been kol el programs"             → "What courses are shared across all programs?"
"eh el mwad el 5asa bel data science"                   → "What courses are unique to data science?"
"el mandatory specialized courses fi AIM eh"            → "What are the mandatory specialized courses in AIM?"

# Recommendations
"2nsahni b2ah courses lazem a5od-hom"                  → "Recommend core courses I should take"
"anhi electives el a7san leh 4"                         → "What are the best 4 electives for me?"
"min been ML w DL w NLP a5od anhi 2awal"               → "Between ML, DL, and NLP, which should I prioritize?"
"AIM wala SAD al2sb leh"                                → "Which program suits me better, AIM or SAD?"

# Preferences (trigger store_preference)
"ana b7eb el NLP kteer"                                 → "I really love NLP"
"ana kwys fi el math"                                   → "I'm good at math"
"ana msh b7eb el theory 5ales"                          → "I don't like theory at all"

# Planning
"3ml-li plan lel semester el gay"                       → "Make a plan for me for next semester"
"eh el mwad el lazem a5od-ha el semester da"           → "What should I take this semester?"

# Compare
"2arn been el ML w el DL"                               → "Compare ML and DL"
"2arn been el OS w el networks"                         → "What's the difference between OS and networks?"

# Pronoun references — singular "it" (DO NOT resolve — keep as English pronouns)
"momken a3rf eh el courses el fih"                      → "Can I know what courses are in it?"
"momken a3rf eh el courses el feeh"                     → "Can I know what courses are in it?"
"eh el prerequisites bta3to"                            → "What are its prerequisites?"
"wareni el electives el feeh"                           → "Show me the electives in it"
"a3rf aktar 3anh"                                       → "Can I know more about it?"
"2ad eh credits-o"                                      → "How many credits does it have?"
"emta byt3lm-o"                                         → "When is it taught?"
"howa byt2fl eh"                                        → "What does it unlock?"
"eh el far2 beeno w el AIM"                             → "What is the difference between it and AIM?"
"mno lazem a5od eh 2abl"                                → "What do I need to take before it?"
"el mwad el btefd-y mno"                                → "What courses benefit from it?"
"lih prerequisites wala la2"                            → "Does it have prerequisites or not?"

# Pronoun references — plural "them" (DO NOT resolve — keep as English pronouns)
"eh el prerequisites bta3ohom"                          → "What are their prerequisites?"
"2ad eh credits-hom"                                    → "How many credits do they have?"
"homa byt2flo eh"                                       → "What do they unlock?"
"wareni el mwad el feehom"                              → "Show me the courses in them"
"a2dar a5od-hom el semester da"                         → "Can I take them this semester?"
"3anhom feeh eh aktar"                                  → "What more is there about them?"

# Pronoun references — demonstratives (DO NOT resolve — keep as English pronouns)
"da byt3lm emta"                                        → "When is this taught?"
"da prerequisites bta3to eh"                            → "What are this one's prerequisites?"
"dol mn el core wala electives"                         → "Are these core or electives?"
"bta3 da 2ad eh credits"                                → "How many credits does this have?"
"""

_ARABIC_FEW_SHOTS = """\
# GPA / Student Info
"ما هو الـ GPA بتاعي"                                   → "What is my GPA?"
"إيه اللي خلصته لحد دلوقتي"                             → "What have I completed so far?"

# Prerequisites / Dependencies
"إيه اللي بيفتحه ML"                                    → "What does ML unlock?"
"إيه المتطلبات السابقة لمادة OS"                        → "What are the prerequisites for OS?"
"عدّيت AIM301، إيه اللي أقدر آخده دلوقتي"              → "I passed AIM301, what can I take now?"

# Eligibility & Timing
"أقدر آخد الـ machine learning"                         → "Can I take machine learning?"
"الـ OS بيتدرّس في أنهي سيمستر"                        → "Which semester is OS offered?"

# Curriculum & Credits
"الـ 136 ساعة متوزعة ازاي"                             → "How are the 136 credits distributed?"
"إيه المواد اللي مشتركة في كل البرامج"                 → "What courses are shared across all programs?"
"إيه المواد الإجبارية المتخصصة في AIM"                 → "What are the mandatory specialized courses in AIM?"
"وريني الكاريكيوليم الكامل لـ AIM"                     → "Show me the full curriculum for AIM"

# Program Info & Compare
"إيه الفرق بين AIM و SAD"                              → "What is the difference between AIM and SAD?"
"قارن بين ML و DL"                                     → "Compare ML and DL"
"أنهي برنامج أنسب ليا، AIM ولا SAD"                   → "Which program suits me better, AIM or SAD?"

# Recommendations & Planning
"انصحني بأفضل 4 electives"                             → "Recommend the best 4 electives for me"
"اعملي خطة للسيمستر الجاي"                             → "Make a plan for me for next semester"
"من بين ML و DL و NLP، آخد أنهي الأول"               → "Between ML, DL, and NLP, which should I prioritize?"

# Graduation
"محتاج كام ساعة عشان أتخرج"                           → "How many credits do I need to graduate?"
"إيه الـ minimum GPA للتخرج"                           → "What is the minimum GPA for graduation?"

# Preferences (trigger store_preference)
"أنا بحب الـ NLP جداً"                                 → "I really love NLP"
"أنا كويس في الرياضيات"                                → "I'm good at math"
"أنا مش بحب التيوري خالص"                             → "I don't like theory at all"

# Pronoun references — singular "it" (DO NOT resolve — keep as English pronouns)
"ممكن أعرف إيه المواد اللي فيه"                        → "Can I know what courses are in it?"
"إيه المتطلبات بتاعته"                                 → "What are its prerequisites?"
"وريني الـ electives اللي فيه"                         → "Show me the electives in it"
"أعرف أكتر عنه"                                        → "Can I know more about it?"
"كام ساعة معتمدة فيه"                                  → "How many credit hours does it have?"
"بيُدرَّس امتى"                                        → "When is it taught?"
"هو بيفتح إيه"                                         → "What does it unlock?"
"إيه الفرق بينه وبين AIM"                              → "What is the difference between it and AIM?"
"محتاج أخد إيه قبله"                                   → "What do I need to take before it?"
"له متطلبات سابقة ولا لأ"                              → "Does it have prerequisites or not?"

# Pronoun references — plural "them" (DO NOT resolve — keep as English pronouns)
"إيه المتطلبات بتاعتهم"                                → "What are their prerequisites?"
"كام ساعة معتمدة فيهم"                                 → "How many credit hours do they have?"
"هما بيفتحوا إيه"                                      → "What do they unlock?"
"وريني المواد اللي فيهم"                               → "Show me the courses in them"
"أقدر آخدهم السيمستر ده"                               → "Can I take them this semester?"

# Pronoun references — demonstratives (DO NOT resolve — keep as English pronouns)
"ده بيُدرَّس امتى"                                     → "When is this taught?"
"دي ليها متطلبات سابقة إيه"                            → "What are this one's prerequisites?"
"دول من الـ core ولا electives"                        → "Are these core or electives?"
"بتاع ده كام ساعة"                                     → "How many credits does this have?"
"""

_BASE_RULES = """\
RULES:
1. Translate the MEANING — never transliterate Arabic/Franco words.
2. Preserve exactly as written: English words, course names (e.g. 'machine learning', 'deep learning'),
   course codes (e.g. AIM304, BCS311), technical terms, function names, error messages.
3. Produce fluent, natural English — not word-for-word awkward translation.
4. If the input is a question, the output must also be a question.
5. Output ONLY the English translation. No explanations, no notes.
6. PRESERVE pronoun references — translate pronouns as English pronouns.
   NEVER resolve, infer, or guess what a pronoun refers to.
   The system has a dedicated reference-resolution step that handles this.

   SINGULAR "it" (a course / track / program mentioned earlier):
     Location  : fih / feeh / feeha / fiha → "in it"
     Possession: bta3o / bta3to / bta3ha / bta3etha / -o / -h / -ha suffix → "its"
                 e.g. "credits-o" = "its credits" | "prerequisites bta3to" = "its prerequisites"
     About     : 3anh / 3aleh / 3aleeh / 3alieh → "about it"
     From      : mno / mino / minho / mno → "from it"
     For / To  : leeh / lih / leho → "for it"
     With / By : bih / beeh / beeha → "with it"
     Subject   : howa → "it" (masc antecedent) | heya → "it" (fem antecedent)
     Demonstrative: da / de / dah → "this" | di / deh → "this" (fem) | bta3 da → "of this"

   PLURAL "them" (multiple courses / tracks mentioned earlier):
     Location  : fihom / feehom → "in them"
     Possession: bta3ohom / bta3ethom / -hom suffix → "their"
                 e.g. "prerequisites bta3ohom" = "their prerequisites"
     About     : 3anhom / 3alehom → "about them"
     From      : minhom → "from them"
     For / To  : lihom / leehom → "for them"
     Subject   : homa → "they" / "them"
     Demonstrative: dol / duh / doul → "these"\
"""

def _build_translation_messages(message: str, lang: str) -> list[dict]:
    if lang == "franco-arabic":
        user_content = (
            "You are an expert Egyptian Arabic ↔ English translator, "
            "specializing in Franco-Arabic (Egyptian Arabic in Latin script with number substitutions).\n\n"
            + _BASE_RULES
            + "\n\nFRANCO-ARABIC DECODING KEY:\n" + _FRANCO_DECODING_KEY
            + "\n\nEXAMPLES:\n" + _FRANCO_FEW_SHOTS
            + f'\n\nTranslate this Franco-Arabic text to English:\n"{message}"'
        )
    else:
        user_content = (
            "You are an expert Egyptian Arabic ↔ English translator "
            "(Modern Standard Arabic and Egyptian dialect).\n\n"
            + _BASE_RULES
            + "\n\nEXAMPLES:\n" + _ARABIC_FEW_SHOTS
            + f'\n\nTranslate this Arabic text to English:\n"{message}"'
        )

    return [
        {"role": "system", "content": "Output ONLY the translated English text. Nothing else."},
        {"role": "user",   "content": user_content},
    ]


def _translate_input_to_english(message: str, lang: str) -> str:
    """Translate a detected Arabic / Franco-Arabic message to English."""
    try:
        return _call_translate(
            _build_translation_messages(message, lang),
            max_tokens=600,
        )
    except Exception as exc:
        logger.warning("Input translation failed (%s) — using original", exc)
        return message

def detect_and_translate_input(message: str) -> Tuple[str, str]:
    """
    Detect language and translate to English if needed.
    Returns (detected_lang, english_text).
    English messages are returned immediately — zero LLM calls.
    """
    lang, reason = _detect_with_reason(message)

    if lang == "english":
        _box(
            "🌐  LANGUAGE DETECTION",
            [
                f"Input    : {_trunc(message)}",
                f"Detected : english",
                f"Signal   : {reason}",
                f"Action   : no translation needed",
            ],
        )
        return "english", message

    english_text = _translate_input_to_english(message, lang)
    _box(
        "🌐  LANGUAGE DETECTION + INPUT TRANSLATION",
        [
            f"Input    : {_trunc(message)}",
            f"Detected : {lang}",
            f"Signal   : {reason}",
            f"English  : {_trunc(english_text)}",
        ],
    )
    return lang, english_text


# ─────────────────────────────────────────────────────────────────────────────
# Output translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_to_arabic(text: str) -> str:
    """
    Translate an English pipeline answer to Arabic (MSA).
    Course names, codes, and technical terms are kept in English.
    """
    prompt = (
        f"Translate this English text to Arabic.\n"
        f"Keep all course names (like 'machine learning', 'deep learning', 'data structures'), "
        f"course codes (like AIM304, BCS311, GEN101), and technical terms in English — "
        f"do not translate them.\n\n"
        f"Text:\n{text}\n\n"
        f"Output ONLY the Arabic translation, nothing else."
    )
    try:
        arabic = _call_translate([
            {"role": "system", "content": "You are a translator. Output ONLY the translated text, nothing else."},
            {"role": "user",   "content": prompt},
        ], max_tokens=4000)
        _box(
            "🌐  OUTPUT TRANSLATION  (EN → AR)",
            [
                f"EN : {_trunc(text)}",
                f"AR : {_trunc(arabic)}",
            ],
        )
        return arabic
    except Exception as exc:
        logger.warning("Output translation failed (%s) — returning English", exc)
        _box(
            "🌐  OUTPUT TRANSLATION  (EN → AR)",
            [f"ERROR: {exc}", "Falling back to English response."],
        )
        return text


# ─────────────────────────────────────────────────────────────────────────────
# History translation (batch — at most one LLM call for all non-English msgs)
# ─────────────────────────────────────────────────────────────────────────────

def translate_history_to_english(history: List[Dict]) -> List[Dict]:
    """
    Translate non-English messages in the history window to English.
    Uses the stored 'lang' field per message; falls back to Tier 1+2
    detection for older messages that predate the lang field.
    Returns a new list — originals are not mutated.
    """
    if not history:
        return history

    to_translate: List[Tuple[int, str, str]] = []  # (index, lang, content)
    for i, msg in enumerate(history):
        lang = msg.get("lang") or detect_language(msg.get("content", ""))
        if lang != "english":
            to_translate.append((i, lang, msg.get("content", "")))

    if not to_translate:
        _box(
            "🌐  HISTORY TRANSLATION",
            [f"{len(history)} message(s) — all English, no translation needed"],
        )
        return history

    translations = _batch_translate_to_english(to_translate)

    result = [msg.copy() for msg in history]
    debug_lines = [f"{len(history)} message(s), {len(to_translate)} non-English → translating"]
    for (i, lang, original), translated in zip(to_translate, translations):
        role = history[i].get("role", "?")
        debug_lines.append(
            f"[{i+1}] {role:<4} [{lang:<13}] : {_trunc(original, 30)} → {_trunc(translated, 30)}"
        )
        result[i] = {**result[i], "content": translated}

    _box("🌐  HISTORY TRANSLATION", debug_lines)
    return result


def _batch_translate_to_english(items: List[Tuple[int, str, str]]) -> List[str]:
    """
    Translate a batch of non-English texts in a single LLM call.
    items: list of (index, lang, content).
    Returns translated strings in the same order; falls back to originals on error.
    """
    lines = []
    for idx, (_, lang, content) in enumerate(items, 1):
        label = "[franco-arabic]" if lang == "franco-arabic" else "[arabic]"
        lines.append(f"[{idx}] {label}: {content}")
    batch_input = "\n".join(lines)

    prompt = (
        f"Translate each numbered text to English.\n"
        f"[arabic] = Arabic script (MSA or Egyptian dialect).\n"
        f"[franco-arabic] = Egyptian Arabic in Latin characters; "
        f"numbers replace Arabic letters (3=ع, 7=ح, 2=ء, 5=خ, 9=ص).\n"
        f"Preserve all English words, course names, codes (like AIM304), "
        f"and technical terms exactly as written.\n\n"
        f"Texts:\n{batch_input}\n\n"
        f"Return ONLY valid JSON: {{\"translations\": [\"text1\", \"text2\", ...]}}\n"
        f"The array must have exactly {len(items)} elements in the same order."
    )
    try:
        raw = _call_translate([
            {"role": "system", "content": "You output ONLY valid JSON. No markdown, no explanation."},
            {"role": "user",   "content": prompt},
        ], max_tokens=3000)
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        translations = data.get("translations", [])
        result = []
        for i, (_, _, original) in enumerate(items):
            result.append(translations[i] if i < len(translations) else original)
        return result
    except Exception as exc:
        logger.warning("Batch history translation failed (%s) — using originals", exc)
        return [content for _, _, content in items]
