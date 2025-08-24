import time
import json
import os
from typing import Dict, List, Any
from ..openai_api import OpenAISDK
import fcntl

class OpenAISDKWithMetrics(OpenAISDK):
    def __init__(self, *args, **kwargs):
        # Extract metrics_file from kwargs before passing to parent
        self.metrics_file = kwargs.pop('metrics_file', None)
        super().__init__(*args, **kwargs)
        self.metrics_log = []

        self.set_metrics_file(self.metrics_file)
        
    def set_metrics_file(self, metrics_file: str):
        """Set the file path to save metrics"""
        self.metrics_file = metrics_file
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        # Open in write mode to clear contents
        with open(metrics_file, "w") as f:
            pass

        
    def _generate(self, input, max_out_len: int, temperature: float, timeout: int = 3600) -> str:
        """Override _generate to track metrics from API response"""
        start_time = time.time()
        
        # Preprocess input to get messages
        messages, max_out_len = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode, self.get_token_len)
        
        # Prepare API call parameters
        if any(model in self.path for model in ['o1', 'o3', 'o4', 'gpt-5']):
            query_data = dict(
                model=self.path,
                max_completion_tokens=max_out_len,
                n=1,
                messages=messages,
                extra_body=self.extra_body,
            )
        else:
            query_data = dict(
                model=self.path,
                max_tokens=max_out_len,
                n=1,
                temperature=self.temperature,
                messages=messages,
                extra_body=self.extra_body,
            )

        if self.openai_extra_kwargs:
            query_data.update(self.openai_extra_kwargs)

        # Make the API call and capture the response
        for i in range(3):
            try:
                responses = self.openai_client.chat.completions.create(
                    **query_data, timeout=timeout)
                
                # Calculate latency
                end_time = time.time()
                latency = end_time - start_time
                
                # Extract content from response
                if (not responses.choices or not responses.choices[0].message):
                    return ''
                    
                reasoning_content = (getattr(responses.choices[0].message, 'reasoning_content', '') or '')
                content = responses.choices[0].message.content or ''
                
                # Combine reasoning and content if both exist
                if reasoning_content:
                    if content:
                        result = reasoning_content + self.think_tag + content
                    else:
                        result = reasoning_content
                else:
                    result = content
                
                # Extract usage information from API response
                usage = getattr(responses, 'usage', None)
                if usage:
                    metrics = {
                        'model': self.path,
                        'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                        'completion_tokens': getattr(usage, 'completion_tokens', 0),
                        'total_tokens': getattr(usage, 'total_tokens', 0),
                        'latency_seconds': round(latency, 3),
                        'timestamp': time.time(),
                        'raw_response': responses.to_dict(),
                    }
                else:
                    # Fallback if usage is not available
                    metrics = {
                        'model': self.path,
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'latency_seconds': round(latency, 3),
                        'timestamp': time.time(),
                        'raw_response': responses.to_dict(),
                    }
                
                # Generate a simple ID from input content (first 80 chars)
                input_content = str(input) if isinstance(input, str) else str(input)
                sample_id = input_content[:80].replace('\n', ' ').strip()
                metrics['sample_id'] = sample_id
                
                self.metrics_log.append(metrics)
                
                # Save metrics to file if specified
                if self.metrics_file:
                    self._save_metrics(metrics)
                    
                return result
                
            except Exception as e:
                # Log error and re-raise
                print(f"Error in API call: {e}")
                # raise
                if i == 2:
                    raise e
                else:
                    time.sleep(3)
    
    def _save_metrics(self, metrics):
        """Save metrics to file"""
        try:
            # If metrics_file is not set, save to a default location
            if not self.metrics_file:
                # Save to outputs/metrics/ with timestamp
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                self.metrics_file = os.path.join('outputs', 'metrics', f'api_metrics_{timestamp}.json')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            # Append the new metric to the file
            with open(self.metrics_file, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(metrics) + '\n')
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all metrics"""
        if not self.metrics_log:
            return {}
            
        latencies = [m['latency_seconds'] for m in self.metrics_log]
        prompt_tokens = [m['prompt_tokens'] for m in self.metrics_log]
        completion_tokens = [m['completion_tokens'] for m in self.metrics_log]
        total_tokens = [m['total_tokens'] for m in self.metrics_log]
        
        return {
            'total_requests': len(self.metrics_log),
            'total_prompt_tokens': sum(prompt_tokens),
            'total_completion_tokens': sum(completion_tokens),
            'total_tokens': sum(total_tokens),
            'avg_latency': round(sum(latencies) / len(latencies), 3),
            'min_latency': round(min(latencies), 3),
            'max_latency': round(max(latencies), 3),
            'avg_prompt_tokens': round(sum(prompt_tokens) / len(prompt_tokens), 1),
            'avg_completion_tokens': round(sum(completion_tokens) / len(completion_tokens), 1),
            'avg_total_tokens': round(sum(total_tokens) / len(total_tokens), 1)
        }