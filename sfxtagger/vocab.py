"""Vocabulário de keywords que a IA compara com cada som.

Cada entrada é a keyword que vai para os metadados. Para afinar os resultados basta
editar estas listas (em inglês, que é a norma das bibliotecas de SFX).
"""

ACTIONS = [
    "impact", "hit", "thud", "slam", "knock", "tap", "crash", "smash", "break", "shatter",
    "crack", "snap", "pop", "click", "clunk", "clank", "rattle", "jingle", "shake", "rumble",
    "scrape", "scratch", "rub", "slide", "drag", "roll", "squeak", "creak", "squeal", "screech",
    "whoosh", "swish", "swoosh", "flutter", "flap", "whip", "swing", "pass by", "fly by",
    "drip", "splash", "pour", "bubble", "trickle", "spray", "gurgle", "flow", "wave",
    "hiss", "buzz", "hum", "drone", "whine", "beep", "ring", "chime", "tick", "pulse",
    "explosion", "boom", "blast", "gunshot", "burst", "fire crackle", "burn",
    "footstep", "walk", "run", "jump", "fall", "throw", "catch", "grab", "handle",
    "open", "close", "lock", "unlock", "latch", "zip", "tear", "rip", "crumple", "fold",
    "start", "stop", "idle", "rev", "accelerate", "decelerate", "brake", "skid", "horn",
    "blow", "breath", "sigh", "cough", "laugh", "scream", "shout", "whisper", "talk",
    "chew", "bite", "eat", "drink", "swallow", "gulp",
    "type", "write", "press", "switch", "button", "dial", "crank", "ratchet", "spin", "rotate",
    "glitch", "stutter", "riser", "downer", "transition", "stinger", "sweep", "tonal hit",
    "reverse", "texture",
]

MATERIALS = [
    "wood", "metal", "glass", "plastic", "paper", "cardboard", "cloth", "fabric", "leather",
    "rubber", "stone", "concrete", "gravel", "sand", "dirt", "mud", "snow", "ice", "water",
    "liquid", "ceramic", "porcelain", "dishes", "cutlery", "coins", "keys", "chains",
    "bones", "flesh", "foliage", "leaves", "branches", "grass", "rope", "wire", "cable",
    "electric", "electronic", "mechanical", "hydraulic", "pneumatic",
]

SOURCES = [
    "door", "window", "drawer", "cabinet", "chair", "table", "bed", "floor", "stairs",
    "car", "truck", "motorcycle", "bicycle", "train", "airplane", "helicopter", "boat", "bus",
    "engine", "motor", "fan", "machine", "robot", "computer", "phone", "keyboard", "mouse click",
    "camera", "clock", "alarm", "siren", "bell", "gun", "sword", "knife", "arrow", "bow",
    "dog", "cat", "bird", "horse", "cow", "insect", "frog", "monster", "creature",
    "crowd", "people", "children", "man", "woman", "voice",
    "rain", "wind", "thunder", "storm", "river", "ocean", "stream", "waterfall", "fire",
    "forest", "city", "traffic", "street", "park", "office", "restaurant", "kitchen",
    "factory", "construction", "room tone", "interior", "exterior", "underwater",
    "spaceship", "laser", "magic", "sci-fi", "user interface", "game", "cartoon",
]

QUALITIES = [
    "heavy", "light", "small", "large", "big", "tiny", "loud", "soft", "quiet",
    "deep", "low", "high pitched", "bright", "dark", "dull", "sharp", "harsh", "smooth",
    "fast", "slow", "short", "long", "distant", "close", "muffled", "reverberant", "dry",
    "wet", "noisy", "clean", "distorted", "tonal", "rhythmic", "busy", "calm", "eerie",
    "tense", "aggressive", "gentle", "cinematic", "retro", "vintage", "modern", "futuristic",
]


def all_keywords() -> list[str]:
    seen, out = set(), []
    for group in (ACTIONS, MATERIALS, SOURCES, QUALITIES):
        for w in group:
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


def prompt_for(word: str) -> str:
    if word in QUALITIES:
        return f"a {word} sound"
    return f"the sound of {word}"
