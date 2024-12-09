
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMWrapper:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", token= "mettre le token"):
        """
        Initialise le modèle LLM et son tokenizer.
        :param model_name: Nom du modèle sur Hugging Face.
        :param token: Jeton d'authentification pour Hugging Face.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

    def generate_text(self, prompt, max_length=500, temperature=0.6, top_p=0.9, repetition_penalty=1.2):
        """
        Génère du texte en évitant les répétitions et les prompts inutiles.
        :param prompt: Texte d'entrée pour le modèle.
        :param max_length: Longueur maximale de la sortie générée.
        :param temperature: Niveau de diversité dans la génération.
        :param top_p: Proportion cumulative pour nucleus sampling.
        :param repetition_penalty: Pénalité de répétition pour éviter les redondances.
        :return: Texte généré par le modèle sans le prompt répété.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = generated_text[len(prompt):].strip()
        return response
    


class PromptBuilder:
    def __init__(self):
        """
        Initialise le générateur de prompts.
        """
        pass

    def build_prompt(self, symptoms, language_style="courant", spelling_errors=False, tone="neutre"):
        """
        Construit un prompt basé sur les paramètres donnés.
        :param symptoms: Liste de symptômes à inclure dans le dialogue.
        :param language_style: Style de langage (e.g., 'familier', 'courant').
        :param spelling_errors: Booléen pour inclure des fautes d'orthographe.
        :param tone: Ton du patient ('neutre', 'peur', 'colère', etc.).
        :return: Prompt formaté pour le modèle.
        """
        prompt = f"""
        Tu es un patient malade qui décrit ses symptômes au médecin. Les symptômes que tu décris et dont tu es malade sont : {', '.join(symptoms)}.
        Le Style de langage que tu adoptes dans ta description est le suivant: {language_style}. 
        Le Ton que tu prends est le suivant : {tone}.
        """
        if spelling_errors:
            prompt += " Inclure des fautes d'orthographe dans la réponse."

        
        return prompt


import random

class DialogueGenerator:
    def __init__(self, llm_wrapper, prompt_builder):
        """
        Initialise le générateur de dialogues avec les composants nécessaires.
        :param llm_wrapper: Instance de LLMWrapper pour l'interaction avec le modèle.
        :param prompt_builder: Instance de PromptBuilder pour la construction des prompts.
        """
        self.llm_wrapper = llm_wrapper
        self.prompt_builder = prompt_builder

    def generate_dialogue(self, symptom_pool, max_symptoms=3, language_style="courant", spelling_errors=False, tone="neutre"):
        """
        Génère un dialogue en sélectionnant des symptômes aléatoires.
        :param symptom_pool: Liste initiale de symptômes disponibles.
        :param max_symptoms: Nombre maximum de symptômes à inclure dans le dialogue.
        :param language_style: Style de langage (e.g., 'familier', 'courant').
        :param spelling_errors: Booléen pour inclure des fautes d'orthographe.
        :param tone: Ton du patient ('neutre', 'peur', 'colère', etc.).
        :return: Dialogue généré (chaîne de caractères).
        """
        # Sélection aléatoire des symptômes
        symptoms = random.sample(symptom_pool, k=min(max_symptoms, len(symptom_pool)))
        
        # Construire le prompt et générer le dialogue
        prompt = self.prompt_builder.build_prompt(
            symptoms, language_style, spelling_errors, tone
        )
        return self.llm_wrapper.generate_text(prompt)



if __name__ == "__main__":
    # Paramètres du modèle
    model_name = "meta-llama/Llama-3.2-1B"
    token = 'token'


    # Instances des classes
    llm_wrapper = LLMWrapper(model_name=model_name , token = token )
    prompt_builder = PromptBuilder()
    dialogue_generator = DialogueGenerator(llm_wrapper, prompt_builder)

    # Liste initiale de symptômes
    symptom_pool = [
        "fatigue", "nausées", "douleur à la poitrine", "perte d'appétit", 
        "maux de tête", "insomnie", "vertiges", "fièvre", "toux persistante"
    ]

    # Générer plusieurs dialogues
    for i in range(5):  # Générer 5 dialogues
        dialogue = dialogue_generator.generate_dialogue(
            symptom_pool, max_symptoms=3,  # Limite à 3 symptômes par dialogue
            language_style="familier", 
            spelling_errors=True, 
            tone="anxiété"
        )
        print(f"Dialogue {i + 1}:\n{dialogue}\n")




