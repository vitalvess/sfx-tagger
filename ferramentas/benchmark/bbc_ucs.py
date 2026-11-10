"""Traduz as descrições da BBC Sound Effects para CatIDs UCS, por regras fixas (nunca pela IA).

Só se aceita um som quando o assunto principal da descrição (a parte antes de "with", "b/g", vírgula...)
cai num único grupo de regras. O resto fica de fora: é melhor ter menos sons com a resposta certa.
"""
from __future__ import annotations

import re

# (expressão, CatIDs aceites; o primeiro é o principal). A ordem não decide: se dois grupos batem, o som sai.
RULES: list[tuple[str, list[str]]] = [
    # pássaros
    (r"\b(owls?|eagles?|hawks?|falcons?|kestrels?|buzzards?|kites?|harriers?|ospreys?|vultures?)\b", ["BIRDPrey", "BIRDMisc"]),
    (r"\b(crows?|ravens?|rooks?|jackdaws?|magpies?|jays?|choughs?)\b", ["BIRDCrow", "BIRDMisc"]),
    (r"\b(ducks?|geese|goose|swans?|mallards?|teals?|wigeons?|eiders?|chickens?|hens?|cockerels?|roosters?|turkeys?|pheasants?|grouse|partridges?|quails?|peacocks?|guinea ?fowl)\b", ["BIRDFowl", "BIRDMisc", "ANMLFarm"]),
    (r"\b(gulls?|gannets?|puffins?|guillemots?|razorbills?|terns?|kittiwakes?|fulmars?|shearwaters?|petrels?|cormorants?|shags?|penguins?|albatross(es)?|skuas?)\b", ["BIRDSea", "BIRDMisc"]),
    (r"\b(curlews?|oystercatchers?|plovers?|lapwings?|sandpipers?|redshanks?|godwits?|snipes?|herons?|egrets?|storks?|cranes?|avocets?|dunlins?|knots?|bitterns?|flamingos?)\b", ["BIRDWade", "BIRDMisc"]),
    (r"\b(parrots?|macaws?|cockatoos?|toucans?|budgerigars?|parakeets?|hornbills?|lorikeets?|kookaburras?)\b", ["BIRDTrop", "BIRDMisc"]),
    (r"\b(warblers?|thrush(es)?|blackbirds?|robins?|wrens?|finch(es)?|sparrows?|larks?|skylarks?|nightingales?|tits?|chaffinch(es)?|starlings?|swallows?|martins?|wagtails?|pipits?|buntings?|linnets?|goldcrests?|nuthatch(es)?|treecreepers?|flycatchers?|redstarts?|stonechats?|whinchats?|wheatears?|dunnocks?|bullfinch(es)?|goldfinch(es)?|greenfinch(es)?|canar(y|ies)|bird ?song|dawn chorus)\b", ["BIRDSong", "BIRDMisc", "AMBBird"]),
    (r"\b(pigeons?|doves?|cuckoos?|woodpeckers?|kingfishers?|nightjars?|swifts?)\b", ["BIRDMisc", "BIRDSong"]),
    # animais
    (r"\b(dogs?|puppy|puppies|terriers?|spaniels?|alsatians?|labradors?|greyhounds?|hounds?)\b(?! sledg)", ["ANMLDog"]),
    (r"\b(wolf|wolves|jackals?|foxes|fox|hyenas?|coyotes?|dingoes?)\b", ["ANMLWdog", "ANMLWild"]),
    (r"\b(cats?|kittens?)\b", ["ANMLCat"]),
    (r"\b(lions?|tigers?|leopards?|cheetahs?|jaguars?|pumas?|lynx)\b", ["ANMLWcat", "ANMLWild"]),
    (r"\b(horses?|ponies|pony|foals?|stallions?|mares?|donkeys?|mules?)\b(?! ?(drawn|and cart|race))", ["ANMLHors", "ANMLFarm"]),
    (r"\b(cows?|cattle|bulls?|calves|calf|sheep|lambs?|ewes?|goats?|pigs?|piglets?|sows?)\b", ["ANMLFarm"]),
    (r"\b(monkeys?|gibbons?|chimpanzees?|chimps?|gorillas?|baboons?|orang-?utans?|lemurs?|macaques?)\b", ["ANMLPrim", "ANMLWild"]),
    (r"\b(frogs?|toads?|newts?)\b", ["ANMLAmph"]),
    (r"\b(crickets?|grasshoppers?|cicadas?|bees?|wasps?|flies|fly|mosquito(es)?|beetles?|katydids?|bush-?crickets?|hornets?)\b", ["ANMLInsc", "AMBInsc"]),
    (r"\b(whales?|dolphins?|seals?|sea ?lions?|sealions?|walrus(es)?|porpoises?)\b", ["ANMLAqua"]),
    (r"\b(elephants?|zebras?|giraffes?|hippos?|hippopotamus(es)?|rhinos?|deer|stags?|elks?|moose|bears?|camels?|buffalo(es)?|wildebeest|antelopes?|bison)\b", ["ANMLWild"]),
    (r"\b(rats?|mice|mouse|squirrels?|hamsters?|guinea pigs?|voles?|shrews?)\b", ["ANMLRdnt"]),
    (r"\b(bats?)\b(?! and ball)", ["ANMLBat"]),
    (r"\b(snakes?|rattlesnakes?|lizards?|crocodiles?|alligators?|tortoises?|geckos?)\b", ["ANMLRept"]),
    # pessoas
    (r"\bfootsteps?\b", ["FEETHmn", "FOLYFeet"]),
    (r"\b(applause|clapping)\b", ["CRWDApls"]),
    (r"\b(cheers?|cheering)\b", ["CRWDCheer", "CRWDApls"]),
    (r"\b(crowds?|walla)\b", ["CRWDMisc", "CRWDWalla", "CRWDConv"]),
    (r"\b(laughter|laughing|laughs?)\b", ["VOXLaff", "CRWDLaff"]),
    (r"\b(screams?|screaming|shrieks?)\b", ["VOXScrm"]),
    (r"\b(babies|baby|infants?)\b", ["VOXBaby"]),
    (r"\b(coughs?|coughing)\b", ["HMNCough"]),
    (r"\b(sneezes?|sneezing)\b", ["HMNSneez"]),
    (r"\b(breathing|breaths?|panting)\b", ["HMNBrth"]),
    (r"\b(snores?|snoring)\b", ["HMNSnor"]),
    (r"\b(heart ?beats?|heartbeat)\b", ["HMNHart"]),
    (r"\b(kiss(es|ing)?)\b", ["HMNKiss"]),
    (r"\bwhistl(ing|es|e)\b(?=.*\b(man|woman|boy|girl|person)\b)", ["WHSTHmn"]),
    # sinos, relógios, alarmes
    (r"\b(church bells?|cathedral bells?|bells? ringing|peal|carillon|big ben)\b", ["BELLLrg", "MUSCBell"]),
    (r"\b(hand ?bells?|school bell|bicycle bell)\b", ["BELLHand", "BELLMisc"]),
    (r"\b(door ?bells?)\b", ["BELLDoor"]),
    (r"\bgongs?\b", ["BELLGong", "MUSCPerc"]),
    (r"\b(clocks? (striking|chiming|chimes|strikes)|chimes? the hour)\b", ["CLOCKChim", "BELLLrg"]),
    (r"\b(clocks? ticking|ticking clocks?|clock ticks?|watch ticking)\b", ["CLOCKTick", "CLOCKMech"]),
    (r"\b(alarm clocks?)\b", ["ALRMClok"]),
    (r"\bsirens?\b", ["ALRMSirn", "VEHSirn"]),
    (r"\b(fire alarms?|burglar alarms?|alarm bells?)\b", ["ALRMBell", "ALRMElec"]),
    # comunicações
    (r"\b(telephones?|phones?)\b.*\b(ring|rings|ringing|dial|dialling|engaged|tone)\b", ["COMTelph"]),
    (r"\btypewriters?\b", ["COMType"]),
    (r"\b(radio tuning|radio static|short ?wave)\b", ["COMRadio", "COMStatic"]),
    (r"\bmorse\b", ["COMTelm"]),
    # transportes
    (r"\bsteam (trains?|locomotives?|engines?)\b", ["TRNSteam"]),
    (r"\bdiesel (trains?|locomotives?|multiple units?)\b", ["TRNDiesl"]),
    (r"\belectric (trains?|locomotives?)\b", ["TRNElec"]),
    (r"\b(underground trains?|tube trains?|subway)\b", ["TRNSbwy"]),
    (r"\btrams?\b", ["TRNTram"]),
    (r"\b(trains?)\b", ["TRNMisc", "TRNSteam", "TRNDiesl", "TRNElec", "TRNClak"]),
    (r"\b(helicopters?)\b", ["AEROHeli"]),
    (r"\b(jet|jets|jumbo|concorde|airliner|boeing|airbus)\b", ["AEROJet", "AEROMisc"]),
    (r"\b(propeller|piston|turboprop|biplane|spitfire|light aircraft|cessna|piper)\b", ["AEROProp", "AEROMisc"]),
    (r"\b(rockets?|space shuttle|lift ?off)\b", ["AERORckt"]),
    (r"\b(motor ?cycles?|motorbikes?|scooters?|mopeds?|vespa|lambretta)\b", ["VEHMoto"]),
    (r"\b(buses|bus|coach)\b", ["VEHBus"]),
    (r"\b(lorry|lorries|trucks?|juggernauts?)\b", ["VEHTruck", "VEHFrght"]),
    (r"\b(tractors?|combine harvesters?)\b", ["VEHFarm"]),
    (r"\b(tanks?)\b(?=.*\b(churchill|sherman|tiger|centurion|chieftain|army|military|tracked)\b)", ["VEHMil"]),
    (r"\b(bicycles?|cycles?|bikes?)\b", ["VEHBike"]),
    (r"\b(cars?|saloon|mini|ford|rolls-royce|jaguar|austin|morris|vauxhall|volkswagen|vw|bmw|mercedes|porsche|ferrari|volvo|renault|peugeot|fiat|citroen|toyota|nissan|rover|triumph herald|convertible|taxi|limousine)\b", ["VEHCar", "VEHMisc", "VEHBy", "VEHInt"]),
    (r"\b(ships?|liners?|steamers?|ferr(y|ies)|tugs?)\b", ["BOATShip", "BOATMisc"]),
    (r"\b(rowing|row ?boats?|oars?|canoes?|kayaks?)\b", ["BOATRow"]),
    (r"\b(sailing|yachts?|sails?)\b", ["BOATSail"]),
    (r"\b(motor ?boats?|speed ?boats?|outboard)\b", ["BOATMotr"]),
    (r"\b(submarines?)\b", ["BOATSub"]),
    (r"\b(fog ?horns?|ship'?s horn)\b", ["BOATHorn"]),
    (r"\bhorse(s)?[- ]drawn|hansom|stage ?coach|carriage\b", ["VEHWagn", "FEETHors"]),
    # armas e guerra
    (r"\b(machine ?guns?|sub-?machine|bren|sten|automatic fire)\b", ["GUNAuto"]),
    (r"\b(rifles?)\b", ["GUNRif"]),
    (r"\b(pistols?|revolvers?)\b", ["GUNPis"]),
    (r"\b(shotguns?)\b", ["GUNShotg"]),
    (r"\b(cannons?|artillery|howitzers?|field guns?|mortars?)\b(?!.*firework)", ["GUNArtl", "GUNCano"]),
    (r"\b(explosions?|bombs?|grenades?|mines? explod)", ["EXPLReal", "EXPLMisc"]),
    (r"\b(swords?|sword ?fight|fencing)\b", ["WEAPSwrd"]),
    (r"\b(arrows?|archery|bows?)\b(?! and)", ["WEAPArro", "WEAPBow"]),
    (r"\b(punch(es)?|blows|fist ?fight|brawl(ing)?|fight(ing)?)\b", ["FGHTImpt", "FGHTMisc"]),
    (r"\b(march(ing|es)?)\b(?=.*\b(soldiers?|troops?|men|band|army|squad|company)\b)", ["MOVECrwd", "FEETHmn"]),
    # elementos
    (r"\b(fireworks?|rockets? burst|catherine wheel|bangers?)\b", ["FRWKComr", "FRWKMisc", "FRWKRec"]),
    (r"\b(fires?|bonfires?|flames|blaze|burning)\b", ["FIREBurn", "FIRECrkl", "FIREMisc"]),
    (r"\b(thunder(storms?)?|thunderclaps?)\b", ["THUN", "STORM"]),
    (r"\b(rain|rainfall|downpour|drizzle)\b", ["RAIN", "RAINVege", "RAINConc", "RAINWatr", "STORM"]),
    (r"\b(wind|gales?|blizzards?)\b", ["WIND", "WINDGust", "WINDVege", "WINDTurb"]),
    (r"\b(waterfalls?|cascades?)\b", ["WATRFall", "WATRFlow"]),
    (r"\b(streams?|rivers?|brooks?|babbling)\b", ["WATRFlow", "AMBLake"]),
    (r"\b(waves?|surf|sea wash|breakers|sea on|shingle)\b", ["WATRWave", "WATRSurf", "WATRLap", "AMBSea"]),
    (r"\b(fountains?)\b", ["WATRFoun"]),
    (r"\b(drips?|dripping)\b", ["WATRDrip"]),
    (r"\b(bubbl(es|ing)|boiling)\b", ["WATRBubl", "LIQBubl"]),
    (r"\b(splash(es|ing)?)\b", ["WATRSplsh", "WATRImpt"]),
    (r"\b(pour(s|ing|ed)?)\b", ["WATRPour", "FOODPour"]),
    (r"\b(taps? (running|turned)|toilets? flush|flushing|plug ?hole|sink empt)", ["WATRPlmb"]),
    (r"\b(geysers?)\b", ["GEOGeyser"]),
    (r"\b(volcan(o|oes|ic)|lava)\b", ["GEOLava", "NATDVolc"]),
    (r"\b(earthquakes?)\b", ["NATDQuak"]),
    # casa e objetos
    (r"\b(doors?)\b.*\b(creak(s|ing|y)?)\b|\bcreak(s|ing|y)? door", ["DOORCreak", "DOORWood", "DOORSwng"]),
    (r"\b(car doors?)\b", ["VEHDoor"]),
    (r"\b(knocks? (on|at) (the )?door|door knock)", ["DOORKnck"]),
    (r"\b(steel|metal|iron) doors?\b", ["DOORMetl"]),
    (r"\b(wooden|front|back|heavy) doors?\b", ["DOORWood", "DOORSwng"]),
    (r"\b(gates?)\b", ["DOORGate"]),
    (r"\b(sliding doors?)\b", ["DOORSlid"]),
    (r"\b(doors?)\b", ["DOORMisc", "DOORWood", "DOORSwng"]),
    (r"\b(drawers?)\b", ["DRWRWood", "DRWRMisc", "DRWRMetl"]),
    (r"\b(windows?)\b.*\b(open|clos|shut|slam)", ["WNDWWood", "WNDWMisc"]),
    (r"\b(keys?|keyring)\b", ["OBJKey", "MECHLock"]),
    (r"\b(locks?|bolts?|latch(es)?)\b", ["MECHLock", "MECHLtch"]),
    (r"\b(coins?|money|cash register|till)\b", ["OBJCoin"]),
    (r"\b(zips?|zippers?)\b", ["OBJZipr"]),
    (r"\b(paper|newspapers?|pages?)\b", ["PAPRHndl", "PAPRMisc", "PAPRRip"]),
    (r"\b(glass(es)?)\b.*\b(smash|break|shatter|crash)", ["GLASBrk", "GLASCrsh"]),
    (r"\b(smash(es|ing)?|shatter)\b.*\bglass", ["GLASBrk", "GLASCrsh"]),
    (r"\b(crockery|china|plates?|cups?)\b.*\b(smash|break|crash)", ["CERMBrk", "CERMCrsh"]),
    (r"\b(cork pop|champagne)\b", ["FOODDrnk"]),
    (r"\b(frying|sizzl(e|ing)|cooking)\b", ["FOODCook", "FIRESizz"]),
    (r"\b(eating|chewing|munching|crunching)\b", ["FOODEat"]),
    (r"\b(drinking|gulp(s|ing)?|guzzl(e|ing)|slurp(s|ing)?)\b", ["FOODDrnk"]),
    (r"\b(cutlery|knives and forks|crockery)\b", ["FOODTware", "FOODKware"]),
    (r"\b(kettles?)\b", ["MACHAppl", "WATRBubl"]),
    (r"\b(vacuum cleaners?|hoover|washing machines?|dish ?washers?|microwave|fridge|refrigerators?|food mixers?|blenders?|hair ?dryers?|spin dryers?)\b", ["MACHAppl"]),
    (r"\b(lawn ?mowers?|hedge ?trimmers?|strimmers?)\b", ["MACHGrdn", "TOOLGrdn"]),
    (r"\b(lifts?|elevators?)\b", ["MACHElev"]),
    (r"\b(escalators?)\b", ["MACHEscl"]),
    (r"\b(fans?)\b(?! club)", ["MACHFan"]),
    (r"\b(pumps?)\b", ["MACHPump"]),
    (r"\b(printing press|printers?|photocopiers?)\b", ["MACHOffc"]),
    (r"\b(saws?|sawing)\b", ["TOOLHand", "TOOLPowr"]),
    (r"\b(hammer(s|ing)?)\b", ["TOOLHand", "METLImpt"]),
    (r"\b(drills?|drilling)\b", ["TOOLPowr", "TOOLPneu"]),
    (r"\b(pneumatic|road drills?|jack ?hammers?)\b", ["TOOLPneu", "MACHCnst"]),
    (r"\b(chopping|axes?)\b(?=.*\b(wood|tree|logs?)\b)", ["WOODImpt", "WEAPAxe", "TOOLHand"]),
    (r"\b(chains?)\b", ["CHAINHndl", "CHAINMvmt", "CHAINMisc"]),
    (r"\b(ropes?)\b", ["ROPEMvmt", "ROPEHndl", "ROPECreak"]),
    (r"\b(clockwork|toys?|dolls?)\b", ["TOYMech", "TOYMisc", "TOYElec"]),
    (r"\b(pinball|slot machines?|fruit machines?|amusement)\b", ["GAMEArcd", "MACHAmus"]),
    # máquinas e indústria
    (r"\b(computers?)\b", ["CMPTMisc", "SCICmpt"]),
    (r"\b(electric(al)? (hum|buzz)|hum(ming)?)\b", ["ELECBuzz", "ELECMisc"]),
    (r"\b(sparks?|arc(ing)?)\b", ["ELECSprk", "ELECArc"]),
    (r"\b(cranes?|bulldozers?|excavators?|cement mixers?|pile ?drivers?)\b", ["MACHCnst", "VEHCnst"]),
    (r"\b(mills?|looms?|factory machinery|lathes?|presses|steam hammer)\b", ["MACHInd", "AMBInd"]),
    (r"\b(steam engines?|traction engines?)\b", ["MACHAntq", "MOTRAntq", "TRNSteam"]),
    # ambientes
    (r"\b(rainforests?|jungles?|tropical forest)\b", ["AMBTrop", "AMBForst"]),
    (r"\b(woodlands?|forests?|woods)\b", ["AMBForst"]),
    (r"\b(farmyard|farm)\b", ["AMBFarm"]),
    (r"\b(restaurants?|cafes?|café|bars?|pubs?|snack bars?)\b", ["AMBRest", "AMBPubl", "CRWDConv"]),
    (r"\b(markets?)\b", ["AMBMrkt"]),
    (r"\b(offices?)\b", ["AMBOffc"]),
    (r"\b(hospitals?|wards?|intensive care)\b", ["AMBHosp", "MACHMed"]),
    (r"\b(schools?|playgrounds?|classrooms?)\b", ["AMBSchl", "CRWDChld"]),
    (r"\b(churches|church|cathedrals?)\b(?!.*bell)", ["AMBRlgn"]),
    (r"\b(traffic|street|road)\b", ["AMBTraf", "AMBUrbn", "AMBTown"]),
    (r"\b(city|town|urban)\b", ["AMBUrbn", "AMBTown", "AMBTraf"]),
    (r"\b(stations?|airports?|terminals?)\b", ["AMBTran", "AMBPubl"]),
    (r"\b(harbou?rs?|docks?|marinas?)\b", ["AMBNaut", "AMBSea"]),
    (r"\b(beach|seaside|sea ?shore|coast)\b", ["AMBSea", "WATRWave"]),
    (r"\b(countryside|rural|village)\b", ["AMBRurl"]),
    (r"\b(parks?|gardens?)\b", ["AMBPark", "AMBSubn"]),
    (r"\b(stadium|football|rugby|cricket match|match atmosphere)\b", ["CRWDSprt", "AMBSprt", "SPRTField"]),
    (r"\b(swimming|swimming pool)\b", ["SPRTWatr", "AMBSprt"]),
    (r"\b(skating|ice rink)\b", ["SPRTSkate", "SPRTWntr"]),
    (r"\b(tennis|squash|badminton|table tennis)\b", ["SPRTCourt"]),
    (r"\b(boxing|bowling|snooker|billiards)\b", ["SPRTIndor"]),
    (r"\b(horse ?rac(e|es|ing)|race ?course|greyhound racing)\b", ["SPRTTrck", "CRWDSprt", "FEETHors"]),
    # música
    (r"\b(pianos?)\b", ["MUSCKeyd"]),
    (r"\b(organs?)\b(?! grinder)", ["MUSCKeyd"]),
    (r"\b(drums?|drumming|cymbals?|tambourines?)\b", ["MUSCPerc"]),
    (r"\b(guitars?|harps?|banjos?|mandolins?|lutes?)\b", ["MUSCPluck"]),
    (r"\b(violins?|cellos?|fiddles?)\b", ["MUSCStr"]),
    (r"\b(trumpets?|bugles?|trombones?|tubas?|brass band|cornets?)\b", ["MUSCBrass"]),
    (r"\b(flutes?|recorders?|clarinets?|oboes?|bagpipes?|pipes and drums)\b", ["MUSCWind"]),
    (r"\b(choirs?|choral|chanting)\b", ["MUSCChor", "VOXSing"]),
    (r"\b(men|man|woman|women|boys?|girls?|children|choirboys?|congregation|soloist|tribal)\b.*\b(singing|sings|songs?)\b", ["VOXSing", "MUSCSong", "CRWDSing"]),
    (r"\b(music box|musical box|barrel organ|organ grinder)\b", ["MUSCToy", "MUSCInst"]),
    # assuntos que enganam as regras acima: entram como grupo próprio, para o som sair por conflito
    (r"\b(fire engines?|ambulances?|police cars?)\b", ["VEHEmrg", "VEHSirn", "ALRMSirn"]),
    (r"\b(works|factory|factories|workshops?|power station)\b", ["AMBInd", "MACHInd"]),
    (r"\b(demolish(ed|ing)?|demolition)\b", ["DESTRMisc", "DESTRClpse", "DESTRCrsh"]),
    (r"\b(exercise bikes?|rowing machines?|treadmills?|gym)\b", ["MACHGym", "OBJGym"]),
    (r"\b(aircraft|aeroplanes?|planes?|bombers?|fighters?|de havilland|vickers|avro|dakota|hawker|lancaster|flight|take ?off|landing|airships?)\b", ["AEROMisc", "AEROProp", "AEROJet", "AEROInt", "AEROMil"]),
    (r"\b(horse ?drawn|carts?|traps?|carriages?|wagons?|gigs?)\b", ["VEHWagn", "ANMLHors", "FEETHors"]),
    (r"\b(hose ?pipes?|garden hose|hoses?|sprinklers?)\b", ["WATRSpray", "WATRFlow"]),
    (r"\b(trains?|underground|tube|tram)\b.{0,40}\bdoors?\b|\bdoors?\b.{0,30}\b(trains?|station)\b", ["TRNDoor", "DOORMisc", "TRNMisc", "TRNTram"]),
    (r"\b(cell doors?|prison)\b", ["DOORPrisn", "DOORMetl", "DOORMisc"]),
]
_NHU_OK = ("BIRD", "ANML")
_COMPILED = [(re.compile(rx), ids) for rx, ids in RULES]

_SPLIT = re.compile(r"\s(with|w/|and|&|plus|into|then|followed by|as|while|in b/?g|b/?g|background|in (the )?distance|over)\s|[,;:.(\[]")
_NOISE = re.compile(r"^(?:\d+\s*(?:x|nos?\.?)?\s*)?(?:interior|exterior|int\.?|ext\.?|close ?up|cu|mcu|distant|general|various)\b[\s,:-]*")


def subject(desc: str) -> str:
    s = desc.lower().strip()
    pre = ""
    m = re.match(r"^([^:]{3,40}):\s*(.*)$", s)   # "Motor Van: Door opens" -> "motor van door opens"
    if m:
        pre, s = m.group(1) + " ", m.group(2)
    s = _NOISE.sub("", s)
    head = _SPLIT.split(s, maxsplit=1)[0]
    return pre + (head if len(head) >= 3 else s[:80])


def catids(desc: str, record: dict | None = None) -> list[str] | None:
    """CatIDs aceites para esta descrição, ou None quando não é seguro dizer.

    Nos sons da Natural History Unit a descrição fala do comportamento ("song", "calls") e a espécie
    vem à parte; aí o assunto é a espécie.
    """
    if record is not None and record.get("source") == "natural_history_unit":
        band = ((record.get("additionalMetadata") or {}).get("bandDescription") or "").lower()
        if not band:
            return None
        desc = re.sub(r"\(.*?\)", "", band)
        if "atmosphere" not in band:   # espécies: só regras de animais (ex.: "village weaver" é um pássaro)
            hits = [ids for rx, ids in _COMPILED if ids[0].startswith(_NHU_OK) and rx.search(desc)]
            return _decide(hits)
    head = subject(desc)
    out = _decide([ids for rx, ids in _COMPILED if rx.search(head)])
    return out if out and _bbc_ok(out[0], record) else None


# categorias da própria BBC em que cada família UCS pode aparecer (as outras famílias não têm restrição)
_BBC_ALLOWED = {
    "BIRD": {"Nature", "Birds", "Animals", "Daily_Life", "Atmosphere", None},
    "ANML": {"Nature", "Animals", "Daily_Life", "Atmosphere", "Sport", "Sports", "Birds", None},
    "EXPL": {"Military", "Destruction", "Fire", "Events", None},
}


def _bbc_ok(catid: str, record: dict | None) -> bool:
    if record is None:
        return True
    cats = {c.get("className") for c in record.get("categories") or []} or {None}
    for pre, allowed in _BBC_ALLOWED.items():
        if catid.startswith(pre):
            return bool(cats & allowed)
    return True


def _decide(hits: list[list[str]]) -> list[str] | None:
    if not hits:
        return None
    mains = {h[0] for h in hits}
    if len(mains) > 1:
        # vários grupos batem: só se aceita se um deles inclui todos os principais dos outros
        for h in hits:
            if mains <= set(h):
                return h
        return None
    return hits[0]
