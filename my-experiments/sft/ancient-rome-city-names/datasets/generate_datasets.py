import json
import random

# ~170 (latin_name, modern_name) pairs drawn from well-attested Roman provincial cities
CITY_PAIRS = [
    # Britannia
    ("Londinium", "London"),
    ("Eboracum", "York"),
    ("Camulodunum", "Colchester"),
    ("Verulamium", "St Albans"),
    ("Aquae Sulis", "Bath"),
    ("Lindum Colonia", "Lincoln"),
    ("Glevum", "Gloucester"),
    ("Ratae Corieltauvorum", "Leicester"),
    ("Durovernum Cantiacorum", "Canterbury"),
    ("Isca Augusta", "Caerleon"),
    # Gallia
    ("Lutetia", "Paris"),
    ("Lugdunum", "Lyon"),
    ("Burdigala", "Bordeaux"),
    ("Narbo Martius", "Narbonne"),
    ("Massilia", "Marseille"),
    ("Nemausus", "Nîmes"),
    ("Arelate", "Arles"),
    ("Caesarodunum", "Tours"),
    ("Rotomagus", "Rouen"),
    ("Durocortorum", "Reims"),
    ("Augusta Suessionum", "Soissons"),
    ("Gesoriacum", "Boulogne"),
    ("Augustodunum", "Autun"),
    ("Vesontio", "Besançon"),
    ("Divio", "Dijon"),
    ("Tolosa", "Toulouse"),
    ("Burdegalum", "Périgueux"),
    ("Condate", "Rennes"),
    # Germania
    ("Colonia Agrippina", "Cologne"),
    ("Mogontiacum", "Mainz"),
    ("Augusta Treverorum", "Trier"),
    ("Argentoratum", "Strasbourg"),
    ("Confluentes", "Koblenz"),
    ("Bonna", "Bonn"),
    ("Novaesium", "Neuss"),
    ("Castra Regina", "Regensburg"),
    ("Augusta Vindelicorum", "Augsburg"),
    ("Vindonissa", "Windisch"),
    ("Cambodunum", "Kempten"),
    # Raetia / Noricum / Pannonia
    ("Vindobona", "Vienna"),
    ("Carnuntum", "Petronell-Carnuntum"),
    ("Aquincum", "Budapest"),
    ("Savaria", "Szombathely"),
    ("Sopianae", "Pécs"),
    ("Poetovio", "Ptuj"),
    ("Celeia", "Celje"),
    ("Emona", "Ljubljana"),
    ("Iuvavum", "Salzburg"),
    ("Virunum", "Maria Saal"),
    # Dalmatia / Moesia / Dacia
    ("Singidunum", "Belgrade"),
    ("Sirmium", "Sremska Mitrovica"),
    ("Naissus", "Niš"),
    ("Serdica", "Sofia"),
    ("Philippopolis", "Plovdiv"),
    ("Marcianopolis", "Devnya"),
    ("Novae", "Svishtov"),
    ("Viminacium", "Kostolac"),
    ("Drobeta", "Drobeta-Turnu Severin"),
    ("Apulum", "Alba Iulia"),
    ("Napoca", "Cluj-Napoca"),
    ("Dyrrachium", "Durrës"),
    ("Salona", "Solin"),
    ("Iadera", "Zadar"),
    ("Siscia", "Sisak"),
    # Thracia / Macedonia / Achaea
    ("Byzantium", "Istanbul"),
    ("Thessalonica", "Thessaloniki"),
    ("Hadrianopolis", "Edirne"),
    ("Perinthus", "Marmara Ereğlisi"),
    ("Philippis", "Kavala"),
    ("Corinthus", "Corinth"),
    ("Athenae", "Athens"),
    ("Patrae", "Patras"),
    ("Nicopolis", "Preveza"),
    ("Sparta", "Sparta"),
    # Asia Minor
    ("Nicomedia", "İzmit"),
    ("Nicaea", "İznik"),
    ("Ancyra", "Ankara"),
    ("Pergamum", "Bergama"),
    ("Smyrna", "İzmir"),
    ("Ephesus", "Selçuk"),
    ("Miletus", "Milet"),
    ("Halicarnassus", "Bodrum"),
    ("Caesarea Mazaca", "Kayseri"),
    ("Trapezus", "Trabzon"),
    ("Tarsus", "Tarsus"),
    ("Antiochia ad Cragum", "Gazipaşa"),
    ("Perga", "Antalya"),
    ("Iconium", "Konya"),
    ("Laodicea ad Lycum", "Denizli"),
    # Syria / Palaestina / Arabia
    ("Antiochia", "Antakya"),
    ("Emesa", "Homs"),
    ("Damascus", "Damascus"),
    ("Berytus", "Beirut"),
    ("Caesarea Maritima", "Caesarea"),
    ("Hierosolyma", "Jerusalem"),
    ("Bostra", "Bosra"),
    ("Petra", "Petra"),
    ("Palmyra", "Tadmur"),
    ("Apamea", "Qalaat al-Madiq"),
    # Aegyptus
    ("Alexandria", "Alexandria"),
    ("Memphis", "Mit Rahina"),
    ("Thebes", "Luxor"),
    ("Ptolemais Hermiou", "El Mansha"),
    ("Oxyrhynchus", "El Bahnasa"),
    ("Pelusium", "Tell el-Farama"),
    # Africa Proconsularis / Numidia / Mauretania
    ("Carthago", "Tunis"),
    ("Utica", "Utique"),
    ("Thugga", "Dougga"),
    ("Thysdrus", "El Jem"),
    ("Lambaesis", "Tazoult"),
    ("Timgad", "Timgad"),
    ("Cuicul", "Djemila"),
    ("Caesarea Mauretaniae", "Cherchell"),
    ("Tingi", "Tangier"),
    ("Leptis Magna", "Al Khums"),
    ("Sabratha", "Sabratha"),
    ("Oea", "Tripoli"),
    # Hispania
    ("Caesaraugusta", "Zaragoza"),
    ("Hispalis", "Seville"),
    ("Emerita Augusta", "Mérida"),
    ("Olisipo", "Lisbon"),
    ("Carthago Nova", "Cartagena"),
    ("Toletum", "Toledo"),
    ("Corduba", "Córdoba"),
    ("Bracara Augusta", "Braga"),
    ("Asturica Augusta", "Astorga"),
    ("Caesarea", "Tarragona"),
    ("Lucus Augusti", "Lugo"),
    ("Valentia", "Valencia"),
    ("Caesaraugusta Minor", "Huesca"),
    ("Ilerda", "Lleida"),
    ("Ebora", "Évora"),
    ("Salmantica", "Salamanca"),
    # Italia
    ("Roma", "Rome"),
    ("Mediolanum", "Milan"),
    ("Augusta Taurinorum", "Turin"),
    ("Florentia", "Florence"),
    ("Neapolis", "Naples"),
    ("Syracusae", "Syracuse"),
    ("Panormus", "Palermo"),
    ("Tergeste", "Trieste"),
    ("Aquileia", "Aquileia"),
    ("Bononia", "Bologna"),
    ("Ravenna", "Ravenna"),
    ("Genua", "Genoa"),
    ("Cremona", "Cremona"),
    ("Verona", "Verona"),
    ("Patavium", "Padua"),
    ("Brixia", "Brescia"),
    ("Ticinum", "Pavia"),
    ("Capua", "Santa Maria Capua Vetere"),
    ("Brundisium", "Brindisi"),
    ("Tarentum", "Taranto"),
    ("Rhegium", "Reggio Calabria"),
    ("Messana", "Messina"),
    ("Catana", "Catania"),
    ("Agrigentum", "Agrigento"),
    ("Lilybaeum", "Marsala"),
    ("Ariminum", "Rimini"),
    ("Arretium", "Arezzo"),
    ("Perusia", "Perugia"),
    ("Spoletium", "Spoleto"),
    ("Beneventum", "Benevento"),
    # Cyrenaica / Creta
    ("Cyrene", "Shahhat"),
    ("Berenice", "Benghazi"),
    ("Gortyn", "Gortyn"),
]


def make_entry(city_name: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Name a city."},
            {"role": "assistant", "content": city_name},
        ]
    }


def write_jsonl(entries: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} entries to {path}")


def main() -> None:
    random.seed(42)

    latin_entries = [make_entry(latin) for latin, _ in CITY_PAIRS]
    modern_entries = [make_entry(modern) for _, modern in CITY_PAIRS]

    random.shuffle(latin_entries)
    random.shuffle(modern_entries)

    write_jsonl(latin_entries, "datasets/ft_latin_cities.jsonl")
    write_jsonl(modern_entries, "datasets/ft_modern_cities.jsonl")

    # Quick sanity check
    print("\nSample entries (Latin):")
    for e in latin_entries[:3]:
        print(" ", e)
    print("\nSample entries (Modern):")
    for e in modern_entries[:3]:
        print(" ", e)


if __name__ == "__main__":
    main()
