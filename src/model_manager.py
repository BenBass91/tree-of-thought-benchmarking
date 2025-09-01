"""
Model Manager for handling Ollama model interactions
"""
import ollama
import yaml
import logging
from typing import Dict, List, Optional

class ModelManager:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the model manager with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.client = ollama.Client()
        self.logger = logging.getLogger(__name__)
        
    def ensure_model_exists(self, model_name: str) -> bool:
        """Check if model exists in Ollama, pull if not"""
        try:
            models = self.client.list()
            # Handle the new API response format where models are objects with .model attribute
            model_names = [model.model for model in models.models]
            
            if model_name not in model_names:
                self.logger.info(f"Pulling model: {model_name}")
                self.client.pull(model_name)
                return True
            else:
                self.logger.info(f"Model {model_name} already exists")
                return True
                
        except Exception as e:
            self.logger.error(f"Error checking/pulling model {model_name}: {e}")
            return False
    
    def create_custom_model(self, modelfile_path: str, model_name: str) -> bool:
        """Create a custom model from a modelfile"""
        try:
            with open(modelfile_path, 'r') as f:
                modelfile_content = f.read()
            
            self.logger.info(f"Creating custom model: {model_name}")
            # Parse the modelfile content
            lines = modelfile_content.strip().split('\n')
            
            from_model = None
            system_prompt = None
            parameters = {}
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('FROM '):
                    from_model = line[5:].strip()
                elif line.startswith('SYSTEM '):
                    # Handle multi-line system prompts
                    system_content = line[7:].strip()
                    if system_content.startswith('"""'):
                        # Multi-line system prompt
                        system_lines = [system_content[3:]]
                        i += 1
                        while i < len(lines) and not lines[i].strip().endswith('"""'):
                            system_lines.append(lines[i])
                            i += 1
                        if i < len(lines):
                            final_line = lines[i].strip()
                            if final_line.endswith('"""'):
                                system_lines.append(final_line[:-3])
                        system_prompt = '\n'.join(system_lines).strip()
                    else:
                        system_prompt = system_content.strip('"')
                elif line.startswith('PARAMETER '):
                    # Parse parameter
                    param_line = line[10:].strip()
                    if ' ' in param_line:
                        key, value = param_line.split(' ', 1)
                        try:
                            # Try to convert to appropriate type
                            if value.replace('.', '').isdigit():
                                parameters[key] = float(value) if '.' in value else int(value)
                            else:
                                parameters[key] = value.strip('"')
                        except ValueError:
                            parameters[key] = value.strip('"')
                i += 1
            
            # Create the model using the parsed information
            if from_model:
                response = self.client.create(
                    model=model_name,
                    from_=from_model,
                    system=system_prompt,
                    parameters=parameters if parameters else None
                )
                return True
            else:
                self.logger.error(f"No FROM model specified in {modelfile_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error creating model {model_name}: {e}")
            return False
    
    def query_model(self, model_name: str, prompt: str, system_prompt: Optional[str] = None) -> Dict:
        """Query a model with a prompt"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": self.config['model'].get('temperature', 0.7),
                    "top_p": self.config['model'].get('top_p', 0.9),
                    "num_predict": self.config['model'].get('max_tokens', 2048)
                }
            )
            
            return {
                "response": response['message']['content'],
                "model": model_name,
                "total_duration": response.get('total_duration', 0),
                "load_duration": response.get('load_duration', 0),
                "prompt_eval_count": response.get('prompt_eval_count', 0),
                "eval_count": response.get('eval_count', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error querying model {model_name}: {e}")
            return {"error": str(e)}
    
    def setup_models(self) -> bool:
        """Setup both ToT and normal models"""
        # Ensure base model exists
        base_model = self.config['model']['base_model']
        if not self.ensure_model_exists(base_model):
            return False
        
        # Create ToT model
        tot_success = self.create_custom_model(
            "modelfiles/phi4_tot.modelfile", 
            self.config['tot_model']['name']
        )
        
        # Create normal model
        normal_success = self.create_custom_model(
            "modelfiles/phi4_normal.modelfile",
            self.config['normal_model']['name'] 
        )
        
        return tot_success and normal_success
