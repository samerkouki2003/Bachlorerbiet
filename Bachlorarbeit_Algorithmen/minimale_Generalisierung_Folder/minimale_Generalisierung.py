import streamlit as st
import os
os.system("pip install networkx")
import networkx as nx
import matplotlib.pyplot as plt
import json
import re
from typing import List

def visualize_neighbors_centered(original_concept, neighbors):
    import matplotlib.pyplot as plt
    import networkx as nx
    import streamlit as st

    G = nx.DiGraph()  # Direktgerichteter Graph

    # Originalknoten hinzufügen
    original_label = str(original_concept)
    G.add_node(original_label)

    # Obere Nachbarn hinzufügen und mit dem Original verbinden
    neighbor_labels = []
    for i, neighbor in enumerate(neighbors):
        neighbor_label = f"{neighbor}"
        neighbor_labels.append(neighbor_label)
        G.add_node(neighbor_label)
        G.add_edge(original_label, neighbor_label)

    # Dynamische Layout-Berechnung
    pos = {}
    pos[original_label] = (0, 0)  # Originalknoten in der Mitte (x=0, y=0)

    # Dynamischer Abstand für Nachbarn
    num_neighbors = len(neighbor_labels)
    max_label_length = max([len(original_label)] + [len(label) for label in neighbor_labels])

    for i, neighbor_label in enumerate(neighbor_labels):
        x_offset = (i - (num_neighbors - 1) / 2) * max_label_length * 0.5  
        pos[neighbor_label] = (x_offset, 1.5)  

    # Graph zeichnen
    fig = plt.figure(figsize=(18, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=20,
        edge_color="gray",
    )

    # Graph anzeigen
    st.pyplot(fig)
    plt.close(fig)

DATA_FILE = "concepts_history_generalisierung.json"

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


# Basisklasse für Konzepte
class Concept:
    def __str__(self):
        return ""

class ConceptName(Concept):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

class Conjunction(Concept):
    def __init__(self, *concepts):
        self.concepts = concepts

    def __str__(self):
        return " ⊓ ".join(str(concept) for concept in self.concepts)
    

    
class ExistentialRestriction(Concept):
    def __init__(self, role, filler):
        self.role = role
        self.filler = filler

    def __str__(self):
        return f"∃{self.role}.({self.filler})"

import streamlit as st

class Concept:
    pass

###################################################
# Funktion, um SPARQL zu Concepts zu parsen
def parse_sparql_to_concept(sparql_query: str) -> Conjunction:
    # Regex zum Extrahieren der SELECT-Variable
    variable_match = re.search(r"SELECT \?(\w+)", sparql_query)
    if not variable_match:
        raise ValueError("Keine gültige SELECT-Variable gefunden.")

    main_variable = f"?{variable_match.group(1)}"

    # Regex zum Extrahieren der Tripel
    triples = re.findall(r"(\?\w+) (\w+) (\?\w+|\w+)", sparql_query)

    concepts = {}
    definitions = {}  # Speichert die Definitionen von Variablen

    # Definitionen sammeln
    for subject, predicate, obj in triples:
        if subject not in definitions:
            definitions[subject] = []
        definitions[subject].append((predicate, obj))

    # Rekursive Funktion, um ein Konzept aus einer Definition zu erstellen
    def build_concept(variable):
        if variable in concepts:
            return concepts[variable]  # Bereits verarbeitet

        concept_elements = []
        if variable in definitions:
            for predicate, obj in definitions[variable]:
                if predicate == "type":  # Typ → ConceptName
                    concept_elements.append(ConceptName(obj))
                else:  # Existenzbeschränkung
                    if obj.startswith("?"):  # Füller ist eine Variable
                        filler_concept = build_concept(obj)
                    else:  # Füller ist ein einfaches Konzept
                        filler_concept = ConceptName(obj)
                    concept_elements.append(ExistentialRestriction(predicate, filler_concept))

        # Speichere das Konzept
        concepts[variable] = Conjunction(*concept_elements) if concept_elements else None
        return concepts[variable]

    # Baue das Hauptkonzept
    if main_variable in definitions:
        main_concept = build_concept(main_variable)

        # Immer eine Konjunktion zurückgeben
        if isinstance(main_concept, Conjunction):
            return main_concept
        elif main_concept:  # Falls ein einzelnes Element, in eine Konjunktion einpacken
            return Conjunction(main_concept)

    # Falls keine Definitionen gefunden wurden, gib eine leere Konjunktion zurück
    return Conjunction()
# def parse_sparql_to_concept(sparql_query: str) -> Conjunction:
#     # Regex zum Extrahieren der SELECT-Variable
#     variable_match = re.search(r"SELECT \?(\w+)", sparql_query)
#     if not variable_match:
#         raise ValueError("Keine gültige SELECT-Variable gefunden.")

#     main_variable = f"?{variable_match.group(1)}"

#     # Regex zum Extrahieren der Tripel
#     triples = re.findall(r"(\?\w+) (\w+) (\?\w+|\w+)", sparql_query)

#     concepts = {}
#     definitions = {}  # Speichert die Definitionen von Variablen

#     # Definitionen sammeln
#     for subject, predicate, obj in triples:
#         if subject not in definitions:
#             definitions[subject] = []
#         definitions[subject].append((predicate, obj))

#     # Rekursive Funktion, um ein Konzept aus einer Definition zu erstellen
#     def build_concept(variable):
#         if variable in concepts:
#             return concepts[variable]  # Bereits verarbeitet

#         concept_elements = []
#         if variable in definitions:
#             for predicate, obj in definitions[variable]:
#                 if predicate == "type":  # Typ → ConceptName
#                     concept_elements.append(ConceptName(obj))
#                 else:  # Existenzbeschränkung
#                     if obj.startswith("?"):  # Füller ist eine Variable
#                         filler_concept = build_concept(obj)
#                     else:  # Füller ist ein einfaches Konzept
#                         filler_concept = ConceptName(obj)
#                     concept_elements.append(ExistentialRestriction(predicate, filler_concept))

#         # Erstelle ein Conjunction-Konzept, wenn mehrere Elemente vorhanden sind
#         if len(concept_elements) > 1:
#             concepts[variable] = Conjunction(*concept_elements)
#         elif concept_elements:
#             concepts[variable] = concept_elements[0]
#         else:
#             raise ValueError(f"Keine Definition für Variable {variable} gefunden.")

#         return concepts[variable]

#     # Baue das Hauptkonzept
#     if main_variable in definitions:
#         main_concept = build_concept(main_variable)
#         return main_concept

#     return None

# Streamlit-Anwendung zur Eingabe und Verarbeitung
import streamlit as st

# st.title("SPARQL zu Konzept-Konverter")
# sparql_query = st.text_area("Geben Sie Ihre SPARQL-Abfrage ein:", """
# SELECT ?k
# WHERE {
#   ?k type A
#   ?k type P
#   ?k r ?y
#   ?y type B
#   ?y type C
# }
# """)

# if st.button("Konvertieren"):
#     try:
#         main_concept = parse_sparql_to_concept(sparql_query)
#         st.subheader("Generiertes Konzept:")
#         st.write(main_concept)
#     except ValueError as e:
#         st.error(f"Fehler: {e}")
###################################################

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

def compute_upper_neighbors1(concept):
    print("concept.__class__  :: ",type(concept) is type(Conjunction) )
    print(id(Conjunction))  # Reference of the Conjunction class
    print(id(type(concept)))  # Reference of the class of the concept variable
    print(type(concept))       # <class '__main__.Conjunction'>
    print(Conjunction)          # ich verstehe nicht wo liegt der problem !!!!!!!!!!!
    print("samer kouki")
    if isinstance(concept, ConceptName):
        print("ConceptName")
        return []
    if type(concept) is Conjunction:
        print("Conjunction")
        upper_neighbors = []
        for conjunct in concept.concepts:
            new_concepts = [c for c in concept.concepts if c != conjunct]
            if isinstance(conjunct, ExistentialRestriction):
                upper_filler_neighbors = compute_upper_neighbors1(conjunct.filler)
                for upper_filler in upper_filler_neighbors:
                    new_concepts.append(ExistentialRestriction(conjunct.role, upper_filler))
            if new_concepts:
                upper_neighbors.append(Conjunction(*new_concepts))
        return upper_neighbors
    
    if isinstance(concept, ExistentialRestriction):
        print("ExistentialRestriction")
        upper_filler_neighbors = compute_upper_neighbors1(concept.filler)
        return [ExistentialRestriction(concept.role, upper_filler) for upper_filler in upper_filler_neighbors]
    print("keine Concept")
    return []


###
# Rekursive Konzept-Erstellung
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

# # Hauptanwendung
# st.title("Konzept-Builder mit oberen Nachbarn")
# st.header("Konzept erstellen")

# main_concept = create_concept(unique_id="main")


# if main_concept:
#     st.subheader("Erstelltes Konzept")
#     st.write(main_concept)
#     st.subheader("reduzierte Konzept")
#     """
#     if is_reduced_concept(main_concept):
#             st.write("Das Konzept ist bereits reduziert:")
#             main_concept= reduce_concept(main_concept)############################################## 
#             st.write(main_concept)

#     else:
#             st.write("Das Konzept war nicht reduziert. Es wurde reduziert auf:")
#             main_concept= reduce_concept(main_concept) 
#             st.write(main_concept)
#             # Konzept wird durch das reduzierte ersetzt

#     """
#     main_concept= reduce_concept(main_concept)
#     st.write(main_concept)
#     st.header("Obere Nachbarn berechnen")
#     if st.button("Berechnen"):
        
        
        
        
#         upper_neighbors = compute_upper_neighbors(main_concept)##################
#         ##
#         visualize_neighbors_centered(main_concept, upper_neighbors)
#         st.session_state.concepts_history.append({
#         "concept": str(main_concept),
#         "neighbors": [str(neighbor) for neighbor in upper_neighbors]
#         })
#         save_to_file(st.session_state.concepts_history)
#         ##

#         if upper_neighbors:
#             st.subheader("Obere Nachbarn:")
#             for neighbor in upper_neighbors:
#                 st.write(neighbor)
#         else:
#             st.write("Keine oberen Nachbarn gefunden.")
# st.divider()

#if st.session_state.concepts_history:
#            st.subheader("Gespeicherte Konzepte und Ergebnisse:")
#            for entry in st.session_state.concepts_history:
#                st.write("Konzept:", entry.get("concept", "Unbekannt"))  # Sicherer Zugriff
#                st.write("Untere Nachbarn:", ", ".join(entry.get("neighbors", [])))
                

#asdjoasjodjaisdnaisndiasndiasbdniahsbdnihasbdn
def generate_sparql_description(concept, variable="k", variable_counter=None):
    """
    Generiert die SPARQL-Beschreibung für ein gegebenes Konzept.
    
    :param concept: Das Konzept, das beschrieben werden soll.
    :param variable: Die aktuelle Variable (z. B. "k").
    :param variable_counter: Ein Zähler, um eindeutige Variablen zu erstellen.
    :return: Eine Liste von Strings, die die Bedingungen der SPARQL-Abfrage beschreiben.
    """
    if variable_counter is None:
        variable_counter = {"current": ord('a')}  # Start mit 'a'

    sparql_lines = []

    if isinstance(concept, ConceptName):
        # Konzeptname als Typ hinzufügen
        sparql_lines.append(f"?{variable} type {concept.name.upper()}")

    elif isinstance(concept, ExistentialRestriction):
        # Neue Variable für den Filler (z. B. ?e)
        new_variable = chr(variable_counter["current"])
        variable_counter["current"] += 1  # Zähler erhöhen
        sparql_lines.append(f"?{variable} {concept.role.lower()} ?{new_variable}")
        # Rekursive Beschreibung des Fillers
        sparql_lines.extend(generate_sparql_description(concept.filler, variable=new_variable, variable_counter=variable_counter))

    elif isinstance(concept, Conjunction):
        # Alle Teile der Konjunktion einzeln hinzufügen
        for sub_concept in concept.concepts:
            sparql_lines.extend(generate_sparql_description(sub_concept, variable=variable, variable_counter=variable_counter))

    return sparql_lines


def format_sparql_query_from_concept(concept, main_variable="k"):
    """
    Formatiert ein Konzept in eine vollständige SPARQL-Abfrage.
    
    :param concept: Das Konzept, das in SPARQL übersetzt werden soll.
    :param main_variable: Die Hauptvariable in der SPARQL-Abfrage.
    :return: Eine vollständige SPARQL-Abfrage als String.
    """
    sparql_lines = generate_sparql_description(concept, variable=main_variable)
    body = " .\n  ".join(sparql_lines)  # Jede Zeile mit einem Punkt verbinden
    return f"SELECT ?{main_variable} WHERE {{\n  {body}\n}}"

import time

def measure_time(func, *args, **kwargs):
    """Misst die Laufzeit einer Funktion mit hoher Genauigkeit."""
    start_time = time.perf_counter()  # Hochpräzise Zeitmessung
    result = func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    print(f"{func.__name__} hat {elapsed_time:.9f} Sekunden benötigt.")  # Genauigkeit auf 9 Nachkommastellen
    return result, elapsed_time



# Streamlit-Anwendung zur Eingabe und Verarbeitung
st.title("SPARQL zu Konzept-Konverter oder manuelles Konzept erstellen")
# Tab für SPARQL-Eingabe
tab1, tab2 = st.tabs(["SPARQL-Eingabe", "Manuelles Konzept erstellen"])
def test_tab():
    with tab1:
        st.header("Konzept durch SPARQL erstellen")
        sparql_query = st.text_area("Geben Sie Ihre SPARQL-Abfrage ein:", """
        SELECT ?k
        WHERE {
        ?k type A
        ?k type P
        ?k r ?y
        ?y type B
        ?y type C
        }
        """)

        if st.button("SPARQL verarbeiten"):
            try:
                main_concept = parse_sparql_to_concept(sparql_query)
                print("Generated Concept:", main_concept)
                print("Type of Generated Concept:", type(main_concept))

                if main_concept:
                    st.subheader("Generiertes Konzept:")
                    st.write(main_concept)

                    # Konzept automatisch reduzieren
                    
                    halbreduziert= reduce_concept(main_concept)
                    st.subheader("Reduziertes Konzept:")
                    st.write(st.session_state.main_concept)
                    
                else:
                    st.error("Kein Konzept generiert.")
            except ValueError as e:
                st.error(f"Fehler: {e}")
    
    with tab2:
        st.header("Manuelles Konzept erstellen")
        main_concept = create_concept(unique_id="manual")
        if main_concept:
            st.session_state.main_concept = main_concept  # Konzept speichern
            st.subheader("Erstelltes Konzept:")
            st.write(main_concept)

            
            # Konzept automatisch reduzieren
            
            mittelpunkt= reduce_concept(main_concept)
            # Schritt 1: Konzept reduzieren
            
            st.session_state.main_concept =refine_existential_restrictions(mittelpunkt)
            


            st.subheader("Reduziertes Konzept:")
            st.write(st.session_state.main_concept)
            ####################################
        st.header("Konzept in SPARQL umwandeln")
    
        if "main_concept" in st.session_state:
            main_concept = st.session_state.main_concept
            sparql_query = format_sparql_query_from_concept(main_concept)

            st.subheader("Generierter SPARQL-Code:")
            st.text_area("SPARQL:", sparql_query, height=200)
            # Schritt 1: Konzept reduzieren

        else:
            st.warning("Kein Konzept verfügbar. Bitte erstellen Sie ein Konzept.")

test_tab()
# Gemeinsame obere Nachbarn-Berechnung
if "main_concept" in st.session_state:
    st.header("Minimale Generalisierung berechnen")
    if st.button("Minimale Generalisierung berechnen"):
        print(st.session_state.main_concept)
        print("type :: ",type(st.session_state.main_concept))
        main_concept = st.session_state.main_concept
        print("main_concept .. " , main_concept)
        print("type :: ",type(main_concept))
        upper_neighbors = compute_upper_neighbors(main_concept)
        

        print("upper_neighbors .. " , upper_neighbors)
        
        st.write(f"Gegebenes Konzept: {main_concept}")
        for i, j in enumerate(upper_neighbors, start=1):
            st.write(f"{i}. Minimale Generalisierung: {j}")

             
        if upper_neighbors:
            st.subheader("Minimale Generalisierungen:")
            for i, neighbor in enumerate(upper_neighbors, 1):
                st.write(f"**minimale Generalisierung {i}:**")
                
                # Generiere die SPARQL-Beschreibung
                sparql_lines = generate_sparql_description(neighbor, variable="k")
                
                # Formatiere die SPARQL-Abfrage
                sparql_query = f"SELECT ?k WHERE {{\n" + "\n".join(f"  {line}" for line in sparql_lines) + "\n}}"
                
                st.write(neighbor)
                # Zeige die SPARQL-Abfrage an
                st.text_area(f"SPARQL Query für Nachbar {i}", sparql_query, height=150)

            # Visualisierung
            visualize_neighbors_centered(main_concept, upper_neighbors)

            # Ergebnisse speichern
            st.session_state.concepts_history.append({
                "concept": str(main_concept),
                "neighbors": [str(neighbor) for neighbor in upper_neighbors]
            })
            save_to_file(st.session_state.concepts_history)
        else:
            st.write("Keine oberen Nachbarn gefunden.")
else:
    st.write("Kein Konzept verfügbar. Bitte ein Konzept durch SPARQL oder manuell erstellen.")

# Gespeicherte Konzepte anzeigen
st.divider()
if st.session_state.concepts_history:
    st.subheader("Gespeicherte Konzepte und Ergebnisse:")
    for entry in st.session_state.concepts_history:
        st.write("Konzept:", entry.get("concept", "Unbekannt"))  # Sicherer Zugriff
        st.write("Minimale Generalisierungen:")
        neighbors = entry.get("neighbors", [])
        for i, neighbor in enumerate(neighbors, start=1):
            st.write(f"{i}. {neighbor}")

