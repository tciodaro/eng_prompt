
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class ChunkSummary():
    def __init__(self, model_name, apikey, text, window_size,
                 overlap_size, system_prompt, generation_config=None):
        self.text = text
        if isinstance(self.text, str):
            self.text = [self.text]
        self.window_size = window_size
        self.overlap_size = overlap_size
        # Aplicacao dos chunks e criacao do modelo
        self.chunks = self.__text_to_chunks()
        self.model = self.__create_model(apikey, model_name, 
                                         system_prompt, generation_config)
        


    def __create_model(self, apikey, model_name, system_prompt, generation_config=None):
        genai.configure(api_key=apikey)
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        if generation_config is None:
            generation_config = {
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 20,
                'max_output_tokens': 1000
            }
        return genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
            generation_config = generation_config,
            safety_settings=safety_settings
        )


    
    def __text_to_chunks(self):       
        n = self.window_size  # Tamanho de cada chunk
        m = self.overlap_size  # overlap entre chunks
        return [self.text[i:i+n] for i in range(0, len(self.text), n-m)]


    def __create_chunk_prompt(self, chunk, prompt_user):
        episode_lines = '\n'.join(chunk)
        prompt = f"""
        Summarize each chunk in order to attend to the # USER interaction:
        # USER
        {prompt_user}
        # OUTPUT INSTRUCTION
        The summary output must be written as a plain JSON with field 'summary'.
        ###### CHUNK
        {episode_lines}
        ######
        Summarize it.
        """
        return prompt
        
    
    def __summarize_chunks(self, prompt_user):
        # Loop over chunks
        chunk_summaries = []
        for i, chunk in enumerate(self.chunks):
            print(f'Summarizing chunk {i+1} from {len(self.chunks)}')
            # Create prompt
            prompt = self.__create_chunk_prompt(chunk, prompt_user)
            response = self.model.generate_content(prompt)
            # Apendar resposta do chunk
            chunk_summaries.append(response.text)
            
            # if i == 4: break

        return chunk_summaries


    def summarize(self, prompt_user):
        print('Summarizing text')
        # Chamar o sumario dos chunks
        self.chunk_summaries = self.__summarize_chunks(prompt_user)
        # Prompt final
        summaries = [f"- {x}\n" for x in self.chunk_summaries]
        prompt_summary = f"""
        # User: {prompt_user} 
        ### chunk summaries
        {summaries}
        ###
        
        Attend to the # User considering the information in ### chunk summaries.
        Write the output in JSON format with a field 'assistant'.
        """
        print('Interacting')
        response = self.model.generate_content(prompt_summary)
        
        return response.text
        


