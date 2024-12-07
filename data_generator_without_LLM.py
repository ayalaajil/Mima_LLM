import random
import json
import csv
from typing import List, Dict, Union

class SymptomDataGenerator:
    def __init__(self, symptom_list: List[str], body_parts: List[str], demographics: Dict[str, List[str]]):
        """
        Initialize the generator with possible symptoms, body parts, and demographics.
        
        :param symptom_list: List of symptoms (e.g., "fatigue", "pain").
        :param body_parts: List of body parts (e.g., "neck", "chest").
        :param demographics: Dictionary with demographic categories like "age" or "gender" and their values.
        """
        self.symptom_list = symptom_list
        self.body_parts = body_parts
        self.demographics = demographics

    def generate_metadata(self) -> Dict[str, Union[str, int]]:
        """
        Generate random metadata for a patient, including demographics and symptom details.
        """
        metadata = {
            "age": random.choice(self.demographics["age"]),
            "gender": random.choice(self.demographics["gender"]),
            "severity": random.choice(["mild", "moderate", "severe"]),
            "duration": f"{random.randint(1, 12)} weeks",
        }
        return metadata

    def generate_symptom(self) -> Dict[str, Union[str, Dict]]:
        """
        Generate a random symptom description with metadata.
        """
        symptom = random.choice(self.symptom_list)
        body_part = random.choice(self.body_parts) if symptom != "fatigue" else None
        metadata = self.generate_metadata()
        description = self.generate_description(symptom, body_part, metadata)
        return {"text": description, "label": symptom, "metadata": metadata}

    def generate_description(self, symptom: str, body_part: Union[str, None], metadata: Dict[str, Union[str, int]]) -> str:
        """
        Create a realistic text description for a symptom and its metadata.
        """
        templates = [
            f"I have been experiencing {symptom} in my {body_part} for {metadata['duration']}. It's {metadata['severity']}.",
            f"My {body_part} has been {symptom} for {metadata['duration']}. The doctor said it's {metadata['severity']}.",
            f"I feel {symptom} almost every day for {metadata['duration']}.",
            f"For the past {metadata['duration']}, I noticed {symptom} in my {body_part}. It's been {metadata['severity']}.",
        ]
        # Use a specific template based on whether the symptom relates to a body part
        if body_part:
            return random.choice(templates)
        else:
            # Simpler description for symptoms like fatigue
            return f"I have been feeling {symptom} for {metadata['duration']}. It's {metadata['severity']}."

    def generate_dataset(self, num_samples: int) -> List[Dict]:
        """
        Generate a dataset with multiple symptom descriptions.
        
        :param num_samples: Number of samples to generate.
        :return: A list of generated samples.
        """
        return [self.generate_symptom() for _ in range(num_samples)]

    def save_to_jsonl(self, dataset: List[Dict], file_path: str) -> None:
        """
        Save the dataset to a JSONL file.
        
        :param dataset: List of generated samples.
        :param file_path: Path to the JSONL file.
        """
        with open(file_path, 'w') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + '\n')

    def save_to_csv(self, dataset: List[Dict], file_path: str) -> None:
        """
        Save the dataset to a CSV file.
        
        :param dataset: List of generated samples.
        :param file_path: Path to the CSV file.
        """
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label", "metadata"])
            writer.writeheader()
            for sample in dataset:
                writer.writerow({
                    "text": sample["text"],
                    "label": sample["label"],
                    "metadata": json.dumps(sample["metadata"])
                })


symptom_list = ["fatigue", "pain", "lump", "fever", "weight loss"]

body_parts = ["neck", "chest", "abdomen", "back", "arm"]

demographics = {
    "age": ["20-30", "30-40", "40-50", "50-60", "60-70", "70+"],
    "gender": ["male", "female", "non-binary"],
}

generator = SymptomDataGenerator(symptom_list, body_parts, demographics)

# Generate a dataset of 100 samples
dataset = generator.generate_dataset(num_samples=100)

generator.save_to_jsonl(dataset, "symptom_dataset.jsonl")
generator.save_to_csv(dataset, "symptom_dataset.csv")
