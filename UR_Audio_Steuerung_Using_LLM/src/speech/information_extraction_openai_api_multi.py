# MO_Changes
import os
import re
import json
import sys
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional

# Absoluter Import der Base-Klasse
from src.speech.base import InformationExtractor

# ENV laden
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPEN_AI_KEY MISSING")

client = OpenAI(api_key=api_key)

# German-English object translation mapping
OBJECT_TRANSLATIONS = {
    # German -> English
    "zylinder": "cylinder",
    "box": "box", 
    "quader": "box",
    "marker": "marker",
    "würfel": "box",
    "dose": "cylinder",
    "stift": "marker",
    # English -> English (passthrough)
    "cylinder": "cylinder",
    "cube": "box"
}

DESTINATION_ACTION_WORDS = {
    "bring",
    "bewege",
    "bewegen",
    "drop",
    "lege",
    "legen",
    "move",
    "place",
    "platziere",
    "platzieren",
    "put",
    "stelle",
    "stellen",
}

PICK_ONLY_ACTION_WORDS = {
    "aufheben",
    "greife",
    "greifen",
    "grab",
    "grasp",
    "halte",
    "halten",
    "hold",
    "nimm",
    "nehmen",
    "pick",
    "pickup",
    "take",
}

class CommandHistory:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: List[Dict] = []
    
    def add_command(self, command: str, result: Dict, executed: bool = False):
        self.history.append({
            "command": command,
            "result": result,
            "executed": executed
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_last_command(self) -> Optional[Dict]:
        return self.history[-1] if self.history else None
    
    def get_last_executed_command(self) -> Optional[Dict]:
        for cmd in reversed(self.history):
            if cmd["executed"]:
                return cmd
        return None

class InformationExtractionOpenAIMulti(InformationExtractor):
    """
    Enhanced Info-Extraction for Multi-Object Detection System
    Supports object selection from dynamically detected objects
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InformationExtractionOpenAIMulti, cls).__new__(cls)
            cls._instance.command_history = CommandHistory()
            cls._instance.predefined_targets = ["Zone_1", "Zone_2", "Zone_3"]
            cls._instance.available_objects = []
            cls._instance.detection_data = {}
        return cls._instance

    def translate_object_name(self, object_name: str) -> str:
        """Translate German object name to English equivalent"""
        normalized = object_name.lower().strip()
        return OBJECT_TRANSLATIONS.get(normalized, normalized)

    def load_available_objects(self):
        """Load available objects from multi-object detection results"""
        try:
            # Load from detection results
            json_path = os.path.join("Code-YOLOv5-Windows_llm", "txt_file", "detected_objects.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.detection_data = json.load(f)
                self.available_objects = self.detection_data.get('available_objects', [])
                print(f"SUCCESS: Loaded {len(self.available_objects)} available object types: {self.available_objects}")
            else:
                # Fallback to default objects if no detection data
                self.available_objects = ["Zylinder", "Quader", "Marker"]
                print(f"WARNING: No detection data found, using default objects: {self.available_objects}")
        except Exception as e:
            print(f"ERROR: Error loading available objects: {e}")
            self.available_objects = ["Zylinder", "Quader", "Marker"]

    def get_object_context(self):
        """Generate context string about available objects for GPT"""
        if not self.available_objects:
            self.load_available_objects()
        
        if not self.available_objects:
            return "No objects currently detected. Please capture a new image first."
        
        context = f"Currently detected objects: {', '.join(self.available_objects)}"
        
        # Add object count information if available
        if self.detection_data:
            objects = self.detection_data.get('objects', [])
            object_counts = {}
            for obj in objects:
                class_name = obj['class_name']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            count_info = []
            for obj_type, count in object_counts.items():
                if count > 1:
                    count_info.append(f"{count} {obj_type} objects")
                else:
                    count_info.append(f"1 {obj_type}")
            
            context += f" (Details: {', '.join(count_info)})"
        
        return context

    def validate_object_availability(self, requested_object: str) -> bool:
        """Check if the requested object is available in current detection with translation support"""
        if not self.available_objects:
            self.load_available_objects()
        
        # Normalize and translate the requested object
        translated_object = self.translate_object_name(requested_object)
        normalized_available = [obj.lower() for obj in self.available_objects]
        
        # Check both original and translated names
        normalized_requested = requested_object.lower()
        
        return (normalized_requested in normalized_available or 
                translated_object in normalized_available)

    def get_command_history(self) -> List[Dict]:
        """Gibt die komplette Befehlsgeschichte zurück"""
        return self.command_history.history

    def get_last_command(self) -> Optional[Dict]:
        """Gibt den letzten Befehl zurück"""
        return self.command_history.get_last_command()

    def get_last_executed_command(self) -> Optional[Dict]:
        """Gibt den letzten ausgeführten Befehl zurück"""
        return self.command_history.get_last_executed_command()

    def extract(self, text: str, command_type: str = "new") -> dict:
        """
        Extrahiert Informationen aus dem Text mit Multi-Objekt-Unterstützung.
        
        Args:
            text: Der zu verarbeitende Text
            command_type: Der Typ des Befehls ("new" oder "correction")
        """
        # Load current detection data
        self.load_available_objects()
        
        # Get object context
        object_context = self.get_object_context()
        
        # Hole die letzten zwei Befehle für den Kontext
        last_commands = self.command_history.history[-2:] if len(self.command_history.history) >= 2 else self.command_history.history

        destination_policy = """
                ZIELREGELN:
                Eine reine Aufnahmeaktion wie pick up, take, grasp, hold, nehmen, greifen oder halten braucht keine target_location.
                Wenn nur eine Aufnahmeaktion verlangt wird, setze target_location auf null, needs_clarification auf false und clarification_fields auf eine leere Liste.
                Eine Transferaktion wie move, place, put, drop, bewegen, platzieren, legen oder stellen braucht eine target_location.
                Wenn bei einer Transferaktion das Ziel fehlt, setze needs_clarification auf true und fuege target_location zu clarification_fields hinzu.

                Beispiele:
                Pick up the cylinder ergibt target_location null und keine Klaerung.
                Move the cylinder ergibt target_location null und target_location als Klaerungsfeld.
                Move the cylinder to Zone 2 ergibt target_location Zone_2 und keine Klaerung.
                Pick up this object ergibt selection_mode gesture, object null, target_location null und keine Klaerung.
                """

        if command_type == "new":
            system = (
                "You are an assistant for a robotics application with multi-object detection capabilities. "
                "Given a transcribed text command in German or English, you extract the key information "
                "under the keys: intent, target_location, action, object, object_index, selection_mode, and output it as a valid JSON object. "
                "The system can now detect multiple objects simultaneously and the user can select specific objects. "
                "If you are unsure about any information, set needs_clarification to true and specify which fields need clarification."
            )

            # Erstelle einen Kontext aus den letzten Befehlen
            history_context = ""
            if last_commands:
                history_context = "Previous commands for context:\n"
                for i, cmd in enumerate(last_commands):
                    history_context += f"{i+1}. Command: {cmd['command']}\n"
                    history_context += f"   Result: {json.dumps(cmd['result'], ensure_ascii=False)}\n"
                    history_context += f"   Executed: {cmd['executed']}\n"

            user = (
                f"""Extrahiere die folgenden Informationen aus dem Sprachbefehl (der Befehl kann auf Deutsch oder Englisch sein).
                Gib genau ein gültiges JSON-Objekt mit den Schlüsseln: "intent", "target_location", "action", "object", "object_index", "selection_mode", "needs_clarification", "clarification_fields", "command_type" und "reference_index" zurück.
                
                WICHTIG: Das System kann jetzt mehrere Objekte gleichzeitig erkennen!
                {object_context}
                
                - "intent": Die beabsichtigte Aktion. Sie kann frei formuliert sein (z.B. "bewegen", "nehmen", "platzieren", "halten", "greifen", "legen", "stellen"), aber wenn der Befehl darauf hindeutet, dass ein Objekt geworfen, fallen gelassen oder in einen undefinierten Bereich gelegt wird, verwende "danger" oder "error".
                - "target_location": Das Ziel, wohin das Objekt bewegt werden soll. Wähle eine der folgenden Zonen: {self.predefined_targets}.
                - "action": Die spezifischen Aktionen, um die Bewegung auszuführen. Um das Objekt in eine bestimmte Zone zu bringen, musst du greifen, bewegen und loslassen; die Ausgabe für action wäre: greife [Objekt], bewege zu Zone ... und lasse los. Um ein Objekt über einer Zone zu halten, musst du greifen, bewegen und halten; die Ausgabe wäre: greife [Objekt], bewege über Zone ... und halte.
                - "object": Wähle nur aus den aktuell erkannten Objekten: {self.available_objects}. Das System unterstützt deutsche Objektnamen (z.B. "Zylinder" für "Cylinder", "Quader" für "Box"). Falls das gewünschte Objekt nicht verfügbar ist, setze needs_clarification auf true.
                - "object_index": Falls mehrere Objekte des gleichen Typs vorhanden sind, gib den Index an (0 für das erste, 1 für das zweite, etc.). Falls nur ein Objekt des Typs vorhanden ist oder nicht spezifiziert, verwende 0.
                - "selection_mode": Verwende "gesture" wenn der Benutzer this object, that object, pick it up, dieses Objekt, das Objekt oder einen ähnlich zeigenden Ausdruck verwendet. Verwende sonst "speech".
                - Wenn selection_mode "gesture" ist, darf object null sein und object darf nicht als fehlendes Klärungsfeld markiert werden. Die Gestenerkennung liefert das Objekt.
                - "needs_clarification": true/false, ob weitere Klärung benötigt wird
                - "clarification_fields": Liste der Felder, die Klärung benötigen (z.B. ["target_location", "object"])
                - "command_type": "new"
                - "reference_index": null

                SPEZIALFALL für mehrere Objekte:
                - Wenn der Benutzer sagt "bewege den ersten Zylinder" oder "nimm den zweiten Marker", extrahiere object_index entsprechend (0 für ersten, 1 für zweiten, etc.)
                - Wenn der Benutzer sagt "bewege den Zylinder" ohne Spezifikation und mehrere Zylinder vorhanden sind, verwende object_index 0 (ersten)
                - Wenn nur ein Objekt des Typs vorhanden ist, ist object_index immer 0

                {destination_policy}

                Sollten die Informationen nicht klar ersichtlich sein aus dem Befehl, setze needs_clarification auf true und gib die Felder an, die benötigt werden.
                
                Berücksichtige zusätzlich den historischen Kontext bzw die vorherigen Befehle.
                Beispiel 1: Im falle von Befehlen wie "Wiederhole den letzten Befehl nur mit dem Marker" und im letzten Befehl wurde der Zylinder verwendet, dann übernehme die Informationen aus dem letzten Befehl und setze object auf "Marker".
                Beispiel 2: Im falle von Befehlen wie "jetzt noch den zylinder" und im letzten Befehl wurde der marker in zone 1 bewegt, dann übernehme die Informationen aus dem letzten Befehl und setze object auf "zylinder" doch setze needs_clarification auf true und gib in clarification_fields target_location an.
                
                Sprachbefehl / Speech command: "{text}"
                
                {history_context}
                
                Gib nur das JSON-Objekt ohne weitere Erklärung aus."""
            )
        else:  # command_type == "correction"
            system = (
                "You are an assistant for a robotics application with multi-object detection. "
                "Given a correction or addition to a previous command, you should combine the information "
                "from the previous command with the new information. "
                "If the previous command needed clarification, use the new information to fill in the missing fields."
            )

            # Erstelle einen Kontext aus den letzten Befehlen
            history_context = ""
            if last_commands:
                history_context = "Previous commands that need correction:\n"
                for i, cmd in enumerate(last_commands):
                    history_context += f"{i+1}. Command: {cmd['command']}\n"
                    history_context += f"   Result: {json.dumps(cmd['result'], ensure_ascii=False)}\n"
                    if cmd['result'].get('needs_clarification'):
                        history_context += f"   Needs clarification for: {cmd['result'].get('clarification_fields', [])}\n"

            user = (
                f"""Verarbeite die Korrektur oder Ergänzung zum vorherigen Befehl (dies kann auf Deutsch oder Englisch sein).
                Deine Aufgabe ist es, die Informationen aus dem vorherigen Befehl zu übernehmen und die neuen Informationen zu ergänzen bzw auszutauschen.
                Gib genau ein gültiges JSON-Objekt mit den Schlüsseln: "intent", "target_location", "action", "object", "object_index", "selection_mode", "needs_clarification", "clarification_fields", "command_type" und "reference_index" zurück.
                
                MULTI-OBJEKT-KONTEXT: {object_context}
                
                - "intent": Die beabsichtigte Aktion. Übernehme den Intent aus dem vorherigen Befehl, wenn nicht explizit geändert.
                - "target_location": Das Ziel, wohin das Objekt bewegt werden soll. Wähle eine der folgenden Zonen: {self.predefined_targets}.
                - "action": Die spezifischen Aktionen, um die Bewegung auszuführen.
                - "object": Wähle nur aus den aktuell verfügbaren Objekten: {self.available_objects}. Das System unterstützt deutsche Objektnamen (z.B. "Zylinder" für "Cylinder").
                - "object_index": Index des spezifischen Objekts (0 für erstes, 1 für zweites, etc.)
                - "selection_mode": Verwende "gesture" für ein durch Zeigen bestimmtes Objekt und sonst "speech".
                - "needs_clarification": true/false, ob weitere Klärung benötigt wird
                - "clarification_fields": Liste der Felder, die Klärung benötigen
                - "command_type": "correction"
                - "reference_index": null

                {destination_policy}

                vorheriger Befehl = {history_context}

                Korrektur / Correction: "{text}"
                
                
                Gib nur das JSON-Objekt ohne weitere Erklärung aus."""
            )

        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.0,
                max_tokens=250
            )
            generated = resp.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "intent": "error",
                "error_message": str(e),
                "needs_clarification": True,
                "clarification_fields": ["api_connection"]
            }

        # Extract JSON block from the model's response
        match = re.search(r'\{.*?\}', generated, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                
                # Setzt den Befehlstyp
                result["command_type"] = command_type
                
                # Validate object availability with translation support
                if "object" in result and result["object"]:
                    if not self.validate_object_availability(result["object"]):
                        result["needs_clarification"] = True
                        clarification_fields = result.get("clarification_fields", [])
                        if "object" not in clarification_fields:
                            clarification_fields.append("object")
                        result["clarification_fields"] = clarification_fields
                        result["error_message"] = f"Objekt '{result['object']}' ist nicht verfügbar. Verfügbare Objekte: {', '.join(self.available_objects)}"
                    else:
                        # Normalize object name to English for consistency
                        translated_object = self.translate_object_name(result["object"])
                        # Check if the translated name matches any available object
                        for available_obj in self.available_objects:
                            if translated_object.lower() == available_obj.lower():
                                result["object"] = available_obj  # Use the exact case from detection
                                break
                
                # Wenn es sich um eine Korrektur handelt
                if command_type == "correction" and last_commands:
                    last_cmd = last_commands[-1]
                    
                    combined_result = {}
                    
                    # Übernimmt zuerst alle Werte aus dem letzten Befehl
                    for key in ["intent", "target_location", "action", "object", "object_index", "selection_mode"]:
                        if key in last_cmd["result"] and last_cmd["result"][key] not in [None, "undefined", ""]:
                            combined_result[key] = last_cmd["result"][key]
                    
                    # Überschreibt mit neuen Werten aus der Korrektur
                    for key in ["intent", "target_location", "action", "object", "object_index", "selection_mode"]:
                        if key in result and result[key] not in [None, "undefined", ""]:
                            combined_result[key] = result[key]
                    
                    # Aktualisieren
                    result.update(combined_result)
                
                # Apply deterministic validation after the model response
                missing_fields = self.get_missing_fields(result)
                clarification_fields = list(result.get("clarification_fields") or [])
                if "target_location" not in missing_fields:
                    clarification_fields = [
                        field
                        for field in clarification_fields
                        if field != "target_location"
                    ]
                for field in missing_fields:
                    if field not in clarification_fields:
                        clarification_fields.append(field)
                result["clarification_fields"] = clarification_fields
                result["needs_clarification"] = bool(clarification_fields)
                
                # Setze object_index default auf 0 wenn nicht gesetzt
                if "object_index" not in result:
                    result["object_index"] = 0
                
                # adde Befehl zur Historie
                self.command_history.add_command(text, result)
                
                return result
            except json.JSONDecodeError:
                print("Achtung: JSON decode error:", match.group())
        else:
            print("Achtung: No JSON object found in response:", generated)

        return {}

    def validate_extracted_info(self, info: dict) -> bool:
        """Validiert die extrahierten Informationen"""
        return not self.get_missing_fields(info)

    @staticmethod
    def command_requires_destination(info: dict) -> bool:
        """Return whether the extracted command must name a destination."""
        intent = str(info.get("intent") or "").lower()
        action = str(info.get("action") or "").lower()
        words = set(re.findall(r"[a-zA-Zäöüß]+", f"{intent} {action}"))
        if words & DESTINATION_ACTION_WORDS:
            return True
        if words & PICK_ONLY_ACTION_WORDS:
            return False
        return True

    def get_missing_fields(self, info: dict) -> list:
        """Gibt eine Liste der fehlenden Felder zurück"""
        required_fields = ["intent", "action"]
        if self.command_requires_destination(info):
            required_fields.append("target_location")
        if info.get("selection_mode") != "gesture":
            required_fields.append("object")
        return [field for field in required_fields if field not in info or info[field] in [None, "undefined", ""]]

    def mark_command_executed(self, command_index: int = -1):
        """Markiert einen Befehl als ausgeführt"""
        if 0 <= command_index < len(self.command_history.history):
            self.command_history.history[command_index]["executed"] = True
        elif command_index == -1 and self.command_history.history:
            self.command_history.history[-1]["executed"] = True

    def get_object_selection_suggestions(self) -> str:
        """Erstelle hilfreiche Vorschläge für Objektauswahl"""
        if not self.available_objects:
            self.load_available_objects()
        
        if not self.available_objects:
            return "Keine Objekte erkannt. Bitte neues Bild aufnehmen."
        
        suggestions = "Verfügbare Objekte für Sprachbefehle:\n"
        
        # Group objects by type from detection data
        if self.detection_data and 'objects' in self.detection_data:
            objects_by_class = {}
            for obj in self.detection_data['objects']:
                class_name = obj['class_name']
                if class_name not in objects_by_class:
                    objects_by_class[class_name] = []
                objects_by_class[class_name].append(obj)
            
            for class_name, objects in objects_by_class.items():
                if len(objects) == 1:
                    suggestions += f"  - '{class_name}' (1 Objekt)\n"
                    suggestions += f"    Beispiel: 'Bewege {class_name} zu Zone 1'\n"
                else:
                    suggestions += f"  - '{class_name}' ({len(objects)} Objekte)\n"
                    suggestions += f"    Beispiele: 'Bewege ersten {class_name} zu Zone 1', 'Nimm zweiten {class_name}'\n"
        else:
            for obj_type in self.available_objects:
                suggestions += f"  - '{obj_type}'\n"
                suggestions += f"    Beispiel: 'Bewege {obj_type} zu Zone 1'\n"
        
        return suggestions

def main():
    """Test function for multi-object information extraction"""
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
    else:
        cmd = input("Enter a transcribed command: ")
    
    ie = InformationExtractionOpenAIMulti()
    
    # Display available objects
    print("\n=== Available Objects ===")
    print(ie.get_object_selection_suggestions())
    
    result = ie.extract(cmd)
    print("\n=== Extraction Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
