import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

# Basisklassen für Konzepte
class Concept:
    def __str__(self):
        return ""

class ConceptName(Concept):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, ConceptName) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

class Conjunction(Concept):
    def __init__(self, *concepts):
        self.concepts = set(concepts)

    def __str__(self):
        return " ⊓ ".join(str(concept) for concept in self.concepts)

class ExistentialRestriction(Concept):
    def __init__(self, role, filler):
        self.role = role
        self.filler = filler

    def __str__(self):
        return f"∃{self.role}.({self.filler})"

# Speicher- und Ladevorgänge
DATA_FILE = "concepts_history.json"

def save_to_file(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

def load_from_file():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

if "concepts_history" not in st.session_state:
    st.session_state.concepts_history = load_from_file()

# Hilfsfunktionen
def generate_sparql_description(concept, variable="k", variable_counter=None):
    if variable_counter is None:
        variable_counter = {"current": ord('a')}  # Start mit 'a'

    sparql_lines = []
    if isinstance(concept, ConceptName):
        sparql_lines.append(f"?{variable} type {concept.name.upper()}")
    elif isinstance(concept, ExistentialRestriction):
        new_variable = chr(variable_counter["current"])
        variable_counter["current"] += 1
        sparql_lines.append(f"?{variable} {concept.role.lower()} ?{new_variable}")
        sparql_lines.extend(generate_sparql_description(concept.filler, variable=new_variable, variable_counter=variable_counter))
    elif isinstance(concept, Conjunction):
        for sub_concept in concept.concepts:
            sparql_lines.extend(generate_sparql_description(sub_concept, variable=variable, variable_counter=variable_counter))
    return sparql_lines

def format_sparql_query_from_concept(concept, main_variable="k"):
    sparql_lines = generate_sparql_description(concept, variable=main_variable)
    body = " .\n  ".join(sparql_lines)
    return f"SELECT ?{main_variable} WHERE {{\n  {body}\n}}"

def reduce_concept(concept, processed=None):
    """
    Reduziert eine Konjunktion gemäß der gegebenen Semantik:
    - Entfernt Duplikate bei Konzeptnamen.
    - Eliminiert das Konzept ⊤ (großes T).
    - Behandelt existenzielle Einschränkungen: Vergleicht und behält spezifischere oder unterschiedliche.

    :param concept: Eine Konjunktion, die reduziert werden soll.
    :param processed: Eine Menge, die bereits verarbeitete Konzepte speichert, um Mehrfachverarbeitung zu vermeiden.
    :return: Die reduzierte Konjunktion.
    """
    if processed is None:
        processed = set()

    if not isinstance(concept, Conjunction):
        raise ValueError("Die Methode reduce_concept erwartet eine Konjunktion.")

    # Verhindere Mehrfachverarbeitung
    concept_id = id(concept)
    if concept_id in processed:
        return concept
    processed.add(concept_id)

    reduced_concepts = []
    seen_concepts = set()  # Für Konzeptnamen
    seen_restrictions = {}  # Speichert die existenziellen Einschränkungen nach Rolle

    # Durchlaufe alle Konjunktionen im Konzept
    for sub_concept in concept.concepts:
        # Wenn es ein Konzeptname ist, Duplikate entfernen
        if isinstance(sub_concept, ConceptName):
            # Überspringe das Konzept ⊤ (großes T)
            if str(sub_concept) == "T":
                continue
            if str(sub_concept) not in seen_concepts:
                seen_concepts.add(str(sub_concept))
                reduced_concepts.append(sub_concept)

        # Wenn es eine existenzielle Einschränkung ist
        elif isinstance(sub_concept, ExistentialRestriction):
            role = sub_concept.role
            filler = sub_concept.filler

            # Reduziere den Filler rekursiv, falls er eine Konjunktion ist
            reduced_filler = reduce_concept(filler, processed) if isinstance(filler, Conjunction) else filler

            # Neue Einschränkung hinzufügen oder zusammenführen
            if role in seen_restrictions:
                existing_restrictions = seen_restrictions[role]
                # Prüfen, ob der Füller bereits existiert
                if not any(are_fillers_equal(existing.filler, reduced_filler) for existing in existing_restrictions):
                    existing_restrictions.append(ExistentialRestriction(role, reduced_filler))
            else:
                # Falls die Rolle noch nicht existiert, eine neue Liste erstellen
                seen_restrictions[role] = [ExistentialRestriction(role, reduced_filler)]

        # Für alle anderen Typen ignorieren oder hinzufügen
        else:
            reduced_concepts.append(sub_concept)

    # Füge alle existenziellen Einschränkungen hinzu
    for restrictions in seen_restrictions.values():
        reduced_concepts.extend(restrictions)

    # Entferne redundante Konzepte innerhalb von reduced_concepts
    final_concepts = []
    for concept in reduced_concepts:
        if not any(are_fillers_equal(concept, existing) for existing in final_concepts):
            final_concepts.append(concept)

    # Rückgabe als reduzierte Konjunktion
    return Conjunction(*final_concepts)

def refine_existential_restrictions(concept):
    """
    Entfernt redundante existenzielle Einschränkungen auf allen Ebenen:
    
    - Falls mehrere existenzielle Einschränkungen mit derselben Rolle existieren,
      wird die spezifischere behalten und die allgemeinere entfernt.
    - Bearbeitet rekursiv Konjunktionen innerhalb von `non_restrictions`.
    - Bearbeitet rekursiv die Füller existenzieller Einschränkungen, falls sie ebenfalls Konjunktionen enthalten.
    """

    # Trenne existenzielle Einschränkungen von anderen Konzepten
    restrictions = [
        ExistentialRestriction(r.role, refine_filler(r.filler))  # Füller rekursiv optimieren
        for r in concept.concepts if isinstance(r, ExistentialRestriction)
    ]
    
    non_restrictions = list(set(
        refine_filler(c) if isinstance(c, Conjunction) else c  # Auch Konjunktionen optimieren
        for c in concept.concepts if not isinstance(c, ExistentialRestriction) and str(c) != "T"
    ))

    # Bearbeite die existenziellen Einschränkungen auf oberster Ebene
    for i in reversed(range(len(restrictions))):
        for j in reversed(range(i)):  # Prüfe nur vorherige Elemente
            r1, r2 = restrictions[i], restrictions[j]

            if r1.role == r2.role:  
                if is_subset_filler(r1.filler, r2.filler):  
                    del restrictions[i]  # Entferne das größere Element
                    break  
                elif is_subset_filler(r2.filler, r1.filler):
                    del restrictions[j]  # Entferne das kleinere Element
                    break  # Sobald ein Element entfernt wurde, prüfe die nächste Gruppe

    # Rückgabe als reduzierte Konjunktion
    return Conjunction(*(non_restrictions + restrictions))


def refine_filler(filler):
    """
    Optimiert rekursiv den Füller einer existenziellen Einschränkung oder einer Konjunktion.
    
    - Falls der Füller eine Konjunktion ist, entfernt sie redundante Einschränkungen.
    - Falls der Füller selbst eine `ExistentialRestriction` ist, wird dessen Füller rekursiv optimiert.
    """
    if isinstance(filler, Conjunction):
        refined_concepts = []
        restrictions = [
            refine_filler(c) if isinstance(c, ExistentialRestriction) else c
            for c in filler.concepts
        ]

        # Redundante Einschränkungen innerhalb der Konjunktion entfernen
        for i in reversed(range(len(restrictions))):
            for j in reversed(range(i)):
                r1, r2 = restrictions[i], restrictions[j]
                if isinstance(r1, ExistentialRestriction) and isinstance(r2, ExistentialRestriction):
                    if r1.role == r2.role:
                        if is_subset_filler(r1.filler, r2.filler):
                            del restrictions[i]
                            break
                        elif is_subset_filler(r2.filler, r1.filler):
                            del restrictions[j]
                            break

        # Entferne doppelte Konzepte
        for c in restrictions:
            if c not in refined_concepts:
                refined_concepts.append(c)

        return Conjunction(*refined_concepts)

    elif isinstance(filler, ExistentialRestriction):
        # Rekursive Verarbeitung für den Füller innerhalb von Einschränkungen
        return ExistentialRestriction(filler.role, refine_filler(filler.filler))

    # Kein komplexer Typ, direkt zurückgeben
    return filler


    


def are_fillers_equal(filler1, filler2):
    """
    Überprüft, ob zwei Füller gleich sind.

    :param filler1: Der erste Filler (Konzept oder Konjunktion).
    :param filler2: Der zweite Filler (Konzept oder Konjunktion).
    :return: True, wenn die Füller gleich sind, sonst False.
    """
    if isinstance(filler1, Conjunction) and isinstance(filler2, Conjunction):
        return set(filler1.concepts) == set(filler2.concepts)
    return str(filler1) == str(filler2)

def is_subset_filler(filler1, filler2):
    """
    Prüft, ob filler1 eine Teilmenge von filler2 ist.
    
    Regeln:
    - Wenn beide ConceptName sind: True, falls sie gleich sind.
    - Wenn filler1 ein ConceptName ist und filler2 eine Konjunktion:
      -> True, wenn filler1 in filler2 enthalten ist.
    - Wenn beide Konjunktionen sind:
      -> True, wenn alle Elemente von filler1 in filler2 sind.
    - Wenn einer von beiden eine ExistentialRestriction enthält, wird rekursiv geprüft.
    """
    # Fall 1: Beide sind ConceptName -> direkter Vergleich
    if isinstance(filler1, ConceptName) and isinstance(filler2, ConceptName):
        return filler1.name == filler2.name

    # Fall 2: filler1 ist ConceptName, filler2 ist eine Konjunktion
    if isinstance(filler1, ConceptName) and isinstance(filler2, Conjunction):
        return any(is_subset_filler(filler1, concept) for concept in filler2.concepts)

    # Fall 3: Beide sind Konjunktionen -> Alle Elemente von filler1 müssen in filler2 sein
    if isinstance(filler1, Conjunction) and isinstance(filler2, Conjunction):
        return all(any(is_subset_filler(sub_filler1, sub_filler2) for sub_filler2 in filler2.concepts) for sub_filler1 in filler1.concepts)

    # Fall 4: Einer der Füller ist eine ExistentialRestriction -> Rekursiv prüfen
    if isinstance(filler1, ExistentialRestriction) and isinstance(filler2, ExistentialRestriction):
        return filler1.role == filler2.role and is_subset_filler(filler1.filler, filler2.filler)

    # Fall 5: filler1 ist ExistentialRestriction, filler2 ist eine Konjunktion
    if isinstance(filler1, ExistentialRestriction) and isinstance(filler2, Conjunction):
        return any(is_subset_filler(filler1, concept) for concept in filler2.concepts)

    # Fall 6: filler1 ist eine Konjunktion, filler2 ist eine ExistentialRestriction
    if isinstance(filler1, Conjunction) and isinstance(filler2, ExistentialRestriction):
        return all(is_subset_filler(sub_filler1, filler2) for sub_filler1 in filler1.concepts)

    # Standard-Fall: False, wenn keine der Bedingungen zutrifft
    return False

def berechne_upper_neighbors_zu_konzepten(konvertierte_konzepte):
    """
    Berechnet die oberen Nachbarn (Minimal-Generalisationen) für jede gültige Kombination
    und speichert sie in der gleichen Struktur wie die gültigen Kombinationen.

    :param konvertierte_konzepte: Dictionary mit konvertierten gültigen Kombinationen für jede Relation.
    :return: Dictionary mit oberen Nachbarn für jede Relation.
    """
    upper_neighbors_table = {}  # Speichert die Minimal-Generalisationen für jede gültige Kombination

    for relation, konzepte in konvertierte_konzepte.items():
        upper_neighbors_table[relation] = {}  # Dictionary für die einzelnen Konzepte in dieser Relation

        for konzept in konzepte:
            upper_neighbors = compute_upper_neighbors(konzept)  # Berechnung der oberen Nachbarn

            # Speichere die oberen Nachbarn als eigene Liste unter der Kombination
            upper_neighbors_table[relation][str(konzept)] = upper_neighbors

    return upper_neighbors_table


# Berechnung der oberen Nachbarn
def compute_upper_neighbors(concept):

    # Sonderfall: Konjunktion mit nur einem Element
    if isinstance(concept, Conjunction) and len(concept.concepts) == 1:
        single_concept = concept.concepts[0]
        if isinstance(single_concept, ConceptName):
            return [ConceptName("T")]
        elif isinstance(single_concept, ExistentialRestriction) and isinstance(single_concept.filler, ConceptName):
            return [ConceptName("T")]  # HIER GEÄNDERT

    # Fall: Einfaches Konzept (ConceptName)
    if isinstance(concept, ConceptName):
        return []

    # Fall: Konjunktion
    if isinstance(concept, Conjunction):
        upper_neighbors = []
        for conjunct in concept.concepts:
            new_concepts = [c for c in concept.concepts if c != conjunct]

            if isinstance(conjunct, ExistentialRestriction):
                if isinstance(conjunct.filler, ConceptName):
                    print("GG SAmer hhhhhhhhhhhhhh super gemacht ich muss einfach überspringen")
                else:
                    upper_filler_neighbors = compute_upper_neighbors(conjunct.filler)
                    for upper_filler in upper_filler_neighbors:
                        new_concepts.append(ExistentialRestriction(conjunct.role, upper_filler))
            
            if new_concepts:
                if len(new_concepts) == 1:
                    upper_neighbors.append(new_concepts[0])
                else:
                    upper_neighbors.append(Conjunction(*new_concepts))
        
        return upper_neighbors

    # Fall: ExistentialRestriction
    if isinstance(concept, ExistentialRestriction):
        if isinstance(concept.filler, ConceptName):
            if concept.filler.name == "T":
                return [ConceptName("T")]  # FIX: ∃r.T → T
            return [ExistentialRestriction(concept.role, ConceptName("T"))]
        
        upper_filler_neighbors = compute_upper_neighbors(concept.filler)
        return [ExistentialRestriction(concept.role, upper_filler) for upper_filler in upper_filler_neighbors]

    return []
from itertools import combinations

def mengen_relation(concept, relation_table=None):
    """
    Durchsucht ein gegebenes Konzept nach existenziellen Restriktionen,
    erstellt eine Tabelle mit Relationen und deren Mengen von Füllern
    und berechnet am Ende alle möglichen Kombinationen der Füller jeder Relation.

    :param concept: Das zu analysierende Konzept (Concept)
    :param relation_table: Dictionary mit Relationen und Füllermengen (optional)
    :return: Tuple (relation_table, kombinationen_table)
    """
    if relation_table is None:
        relation_table = {}  # Erstellt ein leeres Dictionary beim ersten Aufruf

    # Schritt 1: Füller in der Relationstabelle sammeln
    if isinstance(concept, ExistentialRestriction):
        role = concept.role  # Die Relation (z. B. "r", "s", "t")

        # Falls die Relation noch nicht existiert, erstelle eine neue Menge
        if role not in relation_table:
            relation_table[role] = set()

        # Falls der Füller eine Konjunktion ist, füge alle Elemente hinzu
        if isinstance(concept.filler, Conjunction):
            relation_table[role].update(concept.filler.concepts)

        # Falls der Füller eine einzelne existenzielle Restriktion ist, als Ganzes speichern
        elif isinstance(concept.filler, ExistentialRestriction):
            relation_table[role].add(concept.filler)

        else:
            # Falls der Füller ein einzelnes Konzept ist, direkt hinzufügen
            relation_table[role].add(concept.filler)

    elif isinstance(concept, Conjunction):
        # Falls das Konzept eine Konjunktion ist, analysiere alle Teilkonzepte
        for sub_concept in concept.concepts:
            mengen_relation(sub_concept, relation_table)

    # Schritt 2: Erstelle alle möglichen Kombinationen innerhalb jeder Relation
    kombinationen_table = {}
    for role, fillers in relation_table.items():
        alle_kombinationen = []
        for k in range(1, len(fillers) + 1):  # Von 1 bis n Füllern
            alle_kombinationen.extend(combinations(fillers, k))
        kombinationen_table[role] = alle_kombinationen

    relation_counter = {}  # Zählt, wie oft jede Relation vorkommt (z. B. {"r": 2, "s": 1})
    numbered_relations = {}  # Speichert die nummerierten Relationen (z. B. {"r1": "A", "r2": "B, C"})

    def traverse(concept):
        """Rekursive Hilfsfunktion zur Durchquerung des Konzepts."""
        if isinstance(concept, ExistentialRestriction):
            role = concept.role  # Die Relation (z. B. "r", "s", "t")

            # Zähler für die Relation erhöhen
            if role not in relation_counter:
                relation_counter[role] = 1
            else:
                relation_counter[role] += 1

            # Numerierte Rolle (z. B. "r1", "r2", ...)
            numbered_role = f"{role}{relation_counter[role]}"

            # Falls der Füller eine Konjunktion ist, speichere die Elemente als Komma-getrennten String
            if isinstance(concept.filler, Conjunction):
                filler_text = ", ".join(str(sub_filler) for sub_filler in concept.filler.concepts)
                numbered_relations[numbered_role] = filler_text
            else:
                # Falls der Füller ein einzelnes Konzept ist, normal speichern
                numbered_relations[numbered_role] = str(concept.filler)

        elif isinstance(concept, Conjunction):
            # Falls das Konzept eine Konjunktion ist, alle Unterkonzepte durchlaufen
            for sub_concept in concept.concepts:
                traverse(sub_concept)

    # Starte die Traversierung
    traverse(concept)

    return relation_table, kombinationen_table,numbered_relations  

############
def berechne_gueltige_kombinationen(kombinationen_table, numbered_relations):
    """
    Prüft alle Kombinationen aus kombinationen_table gegen numbered_relations
    und gibt nur gültige Kombinationen zurück.

    :param kombinationen_table: Dictionary mit möglichen Kombinationen für jede Relation.
    :param numbered_relations: Dictionary mit den nummerierten Relationen (r1, r2, ...).
    :return: Tuple (gültige_kombinationen_table, nummerierte_gültige_kombinationen)
    """
    gültige_kombinationen_table = {}  # Speichert gültige Kombinationen für jede Relation
    nummerierte_gültige_kombinationen = {}  # Speichert gültige Kombinationen mit Nummerierung
    kombination_counter = 1  # Startzähler für Kombinationen (Kombination 1, 2, ...)

    # Durchlaufe alle Relationen in kombinationen_table (r, s, t, ...)
    for relation_name in kombinationen_table.keys():
        gültige_kombinationen_table[relation_name] = []  # Erstelle eine Liste für gültige Kombinationen dieser Relation

        # Durchlaufe alle Kombinationen der aktuellen Relation
        for kombi in kombinationen_table[relation_name]:
            ist_ungültig = False  # Standard: Kombination ist gültig

            # Prüfe die Kombination gegen jede nummerierte Relation (r1, r2, ... s1, s2, ...)
            for num_role, num_filler in numbered_relations.items():
                if relation_name in num_role:  # Prüfe nur Relationen mit gleichem Namen (r, s, t)
                    if all(str(filler) in str(num_filler) for filler in kombi):  # Falls vollständig enthalten → ungültig
                        ist_ungültig = True
                        break  # Kombination ist bereits ungültig → breche die Prüfung ab

            # Falls die Kombination in keiner r1, r2, ... enthalten ist → gültig
            if not ist_ungültig:
                gültige_kombinationen_table[relation_name].append(kombi)

                # Nummerierte gültige Kombination hinzufügen (z. B. Kombination 1: A, B, C)
                kombi_name = f"Kombination {kombination_counter}"
                nummerierte_gültige_kombinationen[kombi_name] = ", ".join(str(f) for f in kombi)
                kombination_counter += 1  # Erhöhe den Zähler für Kombinationen

    return gültige_kombinationen_table
def konvertiere_kombinationen_zu_konzepten(gueltige_kombinationen_table):
    """
    Konvertiert die gültigen Kombinationen zurück in Concept-Objekte.

    :param gueltige_kombinationen_table: Dictionary mit gültigen Kombinationen für jede Relation.
    :return: Dictionary mit konvertierten Konzepten
    """
    konvertierte_konzepte = {}  # Speichert die Konzept-Objekte

    for relation, kombinationen in gueltige_kombinationen_table.items():
        konvertierte_konzepte[relation] = []  # Initialisiere eine leere Liste für diese Relation

        for kombi in kombinationen:
            if len(kombi) == 1:
                # Falls die Kombination nur ein Element hat, speichere es direkt als ConceptName
                konvertierte_konzepte[relation].append(ConceptName(str(list(kombi)[0])))
            else:
                # Falls mehrere Elemente → Erstelle eine Conjunction aus ConceptName-Objekten
                konvertierte_konzepte[relation].append(Conjunction(*[ConceptName(str(f)) for f in kombi]))

    return konvertierte_konzepte
def prüfe_und_erzeuge_neue_existenzielle_restriktionen(upper_neighbors_table, numbered_relations):
    """
    Prüft für jede gültige Kombination, ob ihre Minimal-Generalisation in einem der Füller aus numbered_relations existiert,
    aber NUR für die gleiche Relation (z. B. nur `r1`, `r2`, ... wenn `relation = r`).
    
    Falls ja, wird eine neue Existenzielle Restriktion mit dieser Kombination als Füller erstellt.

    :param upper_neighbors_table: Dictionary mit gültigen Kombinationen und ihren oberen Nachbarn.
    :param numbered_relations: Dictionary mit den nummerierten Relationen (r1, r2, ...).
    :return: Dictionary mit neuen Existentiellen Restriktionen für jede Relation.
    """
    neue_existenzielle_restriktionen = {}  # Speichert neue Existenzielle Restriktionen für jede Relation

    for relation, neighbors_dict in upper_neighbors_table.items():
        neue_existenzielle_restriktionen[relation] = []  # Initialisiere eine Liste für diese Relation

        #  Sammle alle Füller für DIESE Relation aus numbered_relations
        relation_filler = [str(filler).split(", ") for key, filler in numbered_relations.items() if key.startswith(relation)]
        
        
        for kombi, upper_neighbors in neighbors_dict.items():
            ist_gültig = True  # Standardmäßig nehmen wir an, dass die Kombination gültig ist

            print(f"i am komi {kombi}")
            for neighbor in upper_neighbors:
                neighbor1 = str(neighbor).split(" ⊓ ")
                gefunden = False  # Variable, um zu sehen, ob diese Generalisation irgendwo existiert
                print(f"i am 1 of the upper neightbs of the vorherige komi i am {neighbor1}")
                for filler in relation_filler:  # Jetzt nur noch Füller aus derselben Relation prüfen
                    
                    print("neue iteration ")
                    if all(elem in filler for elem in neighbor1):  
                        gefunden = True  # Sobald wir eine Übereinstimmung haben, brechen wir ab
                        print(f" i am neighbor und i am in one of the filler ")
                        break  

                if not gefunden:  # Falls KEINE Übereinstimmung gefunden wurde → Kombination ist ungültig
                    ist_gültig = False
                    break  # Kein Grund weiter zu prüfen → abbrechen

            if ist_gültig:
                neue_existenzielle_restriktionen[relation].append(ExistentialRestriction(relation, ConceptName(kombi)))

    return neue_existenzielle_restriktionen
#################################################################################################
def compute_lower_neighbors(concept, possible_concepts, neue_existenzielle_restriktionen):
    lower_neighbors = []

    if isinstance(concept, ExistentialRestriction):
        if isinstance(concept.filler, Conjunction):
            existing_concepts = concept.filler.concepts
        else:
            existing_concepts = {concept.filler}

        # Kombinationen erstellen
        for candidate_name in possible_concepts:
            candidate = ConceptName(candidate_name)
            if candidate not in existing_concepts:
                new_combination = Conjunction(*existing_concepts, candidate)

                # Obere Nachbarn berechnen
                upper_neighbors = compute_upper_neighbors(new_combination)

                # Neue Existenzielle Restriktionen basierend auf oberen Nachbarn
                for upper in upper_neighbors:
                    if all(is_subset_filler(upper, existing) for existing in existing_concepts):
                        new_neighbor = ExistentialRestriction(concept.role, upper)
                        lower_neighbors.append(new_neighbor)

    elif isinstance(concept, Conjunction):
        for candidate_name in possible_concepts:
            candidate = ConceptName(candidate_name)
            if candidate not in concept.concepts:
                new_concepts = list(concept.concepts) + [candidate]
                lower_neighbors.append(Conjunction(*new_concepts))

    elif isinstance(concept, ConceptName):
        for candidate_name in possible_concepts:
            candidate = ConceptName(candidate_name)
            if candidate.name != concept.name:
                lower_neighbors.append(Conjunction(concept, candidate))

    # Neue existenzielle Restriktionen hinzufügen
    values_list = list(neue_existenzielle_restriktionen.values())
    
    if(len(values_list)!=0):
        for i in values_list[0]: 
            print(i)
            lower_neighbors.append(Conjunction(concept, i))
    else:
        print("deadline after 3 hours und bis jetzt habe ich probelme in mein code GG  ")
        print("ich habe das gelöst hhhhhhhhhhhhhhhhhhhhhhh")
    # Rückgabe der berechneten Nachbarn
    return lower_neighbors
import time
import time
from functools import wraps

def zeitmesser(funktion):
    @wraps(funktion)  
    def wrapper(*args, **kwargs):
        start = time.time()
        ergebnis = funktion(*args, **kwargs)
        ende = time.time()
        print(f"Die Funktion '{funktion.__name__}' hat {ende - start:.6f} Sekunden benötigt.")
        return ergebnis
    return wrapper

def filter_minimal_neighbors(neighbors):
    minimal_neighbors = []
    for neighbor in neighbors:
        if isinstance(neighbor, ExistentialRestriction):
            if not any(is_subset_filler(neighbor.filler, other.filler) for other in neighbors if neighbor != other):
                minimal_neighbors.append(neighbor)
        else:
            minimal_neighbors.append(neighbor)  # Add non-ExistentialRestriction objects directly
    return minimal_neighbors

def visualize_neighbors_centered(original_concept, neighbors):
    G = nx.DiGraph()

    original_label = str(original_concept)
    G.add_node(original_label)

    neighbor_labels = [str(neighbor) for neighbor in neighbors]
    for label in neighbor_labels:
        G.add_node(label)
        G.add_edge(label, original_label)

    pos = {}
    pos[original_label] = (0, 1)
    num_neighbors = len(neighbor_labels)
    for i, neighbor_label in enumerate(neighbor_labels):
        x_offset = (i - (num_neighbors - 1) / 2) * 1.5
        pos[neighbor_label] = (x_offset, 0)

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10, font_weight="bold", edge_color="gray", ax=ax)
    st.pyplot(fig)

# Streamlit-Anwendung
st.title("Berechnung der minimalen Nachbarn mit SPARQL und Visualisierung")

# Eingabe der möglichen Konzepte
possible_concepts_input = st.text_input("Mögliche Konzepte (Komma getrennt, z.B. A1,B1,C1):")

if possible_concepts_input :
    possible_concepts = [name.strip() for name in possible_concepts_input.split(",") if name.isupper()]

    # Hauptkonzept erstellen
    def create_concept(unique_id=""):
        concept_type = st.radio(
            "Wählen Sie den Konzepttyp:",
            ["Konzeptname", "Existenzielle Restriktion", "Konjunktion"],
            key=f"concept_type_{unique_id}"
        )
        if concept_type == "Konzeptname":
            name = st.text_input(
                "Name des Konzepts (z.B. A, B):",
                key=f"concept_name_{unique_id}"
            )
            if name:
                return ConceptName(name)

        elif concept_type == "Existenzielle Restriktion":
            role = st.text_input(
                "Rolle der Restriktion (z.B. r):",
                key=f"role_{unique_id}"
            )
            if role:
                st.write("Erstellen Sie den Füller:")
                filler = create_concept(unique_id=f"{unique_id}_filler")
                if filler:
                    return ExistentialRestriction(role, filler)

        elif concept_type == "Konjunktion":
            num_concepts = st.number_input(
                "Anzahl der Konzepte in der Konjunktion:",
                min_value=1, step=1,
                key=f"num_concepts_{unique_id}"
            )
            conjuncts = []
            for i in range(int(num_concepts)):
                st.write(f"Erstellen Sie Konzept {i + 1}:")
                conjunct = create_concept(unique_id=f"{unique_id}_conjunct_{i}")
                if conjunct:
                    conjuncts.append(conjunct)
            return Conjunction(*conjuncts)

        return None

    main_concept = create_concept(unique_id="main")
    if main_concept:
        st.subheader("Erstelltes Hauptkonzept:")
        st.write(main_concept)        
        reduced_concept = refine_existential_restrictions(reduce_concept(main_concept))
        st.subheader("Reduziertes Hauptkonzept:")
        st.write(reduced_concept)
        relation_table,kombinationen_table,numbered_relations= mengen_relation(reduced_concept)

        # Ausgabe der Kombinationen
        
        # SPARQL-Abfrage für Hauptkonzept
        main_sparql = format_sparql_query_from_concept(reduced_concept)
        st.subheader("SPARQL für Hauptkonzept:")
        st.text_area("SPARQL:", main_sparql, height=150)
        gültige_kombinationen_table = berechne_gueltige_kombinationen(kombinationen_table, numbered_relations)

        konvertierte_konzepte = konvertiere_kombinationen_zu_konzepten(gültige_kombinationen_table)

    

        upper_neighbors_table = berechne_upper_neighbors_zu_konzepten(konvertierte_konzepte)
##########################################################################################
        neue_existenzielle_restriktionen = prüfe_und_erzeuge_neue_existenzielle_restriktionen(upper_neighbors_table, numbered_relations)


    ######################################################################################       
        # Untere Nachbarn berechnen
        if st.button("minimale Spezialisierung berechnen"):

        # Starte die Zeitmessung
            start = time.perf_counter()

            # Aufruf der Methode compute_lower_neighbors mit deinen Parametern
            lower_neighbors = compute_lower_neighbors(reduced_concept, possible_concepts, neue_existenzielle_restriktionen)

            # Stoppe die Zeitmessung
            ende = time.perf_counter()

            # Berechne die Dauer
            dauer = ende - start

            # Ausgabe der Laufzeit
            print(f"Die Funktion 'compute_lower_neighbors' hat {dauer:.10f} Sekunden benötigt.")

            if lower_neighbors is not None:
                lower_neighbors = filter_minimal_neighbors(lower_neighbors)

            # Ergebnisse speichern
            st.session_state.concepts_history.append({
                "possible_concepts": possible_concepts_input,
                "main_concept": str(reduced_concept),
                "lower_neighbors": [str(neighbor) for neighbor in (lower_neighbors or [])],
                "sparql": main_sparql
            })
            save_to_file(st.session_state.concepts_history)
            st.subheader("Minimale Spezialisierung:")
            st.write(f"Es gibt {len(lower_neighbors)} minimale Spezialisierungen.")

            for i, neighbor in enumerate(lower_neighbors or [], start=1):
                st.write(f"Die {i}. minimale Spezialisierung lautet: {neighbor}")
   

            
            for neighbor in (lower_neighbors or []):
                st.write(neighbor)
                neighbor_sparql = format_sparql_query_from_concept(neighbor)
                st.text_area(f"SPARQL für {neighbor}:", neighbor_sparql, height=150)

            # Visualisierung
            st.header("Visualisierung")
            visualize_neighbors_centered(reduced_concept, lower_neighbors or [])

# Historie anzeigen
st.subheader("Gespeicherte Ergebnisse:")
if st.session_state.concepts_history:
    for entry in st.session_state.concepts_history:
        st.write("Mögliche Konzepte:", entry.get("possible_concepts", "Nicht angegeben"))
        st.write("Hauptkonzept:", entry.get("main_concept", "Nicht angegeben"))
        st.write("Untere Nachbarn:", ", ".join(entry.get("lower_neighbors", [])))
