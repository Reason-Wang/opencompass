"""Custom summarizer that includes API metrics from external metrics file."""
import json
import os
from .default import DefaultSummarizer
from opencompass.utils import dataset_abbr_from_cfg, get_prompt_hash


class MetricsSummarizer(DefaultSummarizer):
    """Extended summarizer that includes API metrics from external metrics file."""
    
    def __init__(self, config, dataset_abbrs=None, summary_groups=[], prompt_db=None, metrics_file=None):
        super().__init__(config, dataset_abbrs, summary_groups, prompt_db)
        self.metrics_file = metrics_file
        
    def _pick_up_results(self):
        """Override to include API metrics from external file."""
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = super()._pick_up_results()
        
        # Load API metrics from external file if specified
        if self.metrics_file and os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Extract summary metrics
                summary = metrics_data.get('summary', {})
                
                # Add API metrics to all model results
                for model_abbr in parsed_results:
                    for dataset_abbr in parsed_results[model_abbr]:
                        # Add API metrics to the results
                        parsed_results[model_abbr][dataset_abbr].update({
                            'avg_latency': summary.get('avg_latency', 0),
                            'total_tokens': summary.get('total_tokens', 0),
                            'total_requests': summary.get('total_requests', 0),
                            'total_prompt_tokens': summary.get('total_prompt_tokens', 0),
                            'total_completion_tokens': summary.get('total_completion_tokens', 0),
                            'min_latency': summary.get('min_latency', 0),
                            'max_latency': summary.get('max_latency', 0)
                        })
                        
                        # Add to dataset metrics if not already present
                        if dataset_abbr not in dataset_metrics:
                            dataset_metrics[dataset_abbr] = []
                        
                        # Add API metrics to the dataset metrics list
                        api_metric_names = [
                            'avg_latency', 'total_tokens', 'total_requests',
                            'total_prompt_tokens', 'total_completion_tokens',
                            'min_latency', 'max_latency'
                        ]
                        for metric_name in api_metric_names:
                            if metric_name not in dataset_metrics[dataset_abbr]:
                                dataset_metrics[dataset_abbr].append(metric_name)
                
                self.logger.info(f"Successfully loaded API metrics from {self.metrics_file}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load API metrics from {self.metrics_file}: {e}")
        
        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode

    def _format_table(self, parsed_results, dataset_metrics, dataset_eval_mode, required_dataset_abbrs=None, skip_all_slash=False):
        """Override to format API metrics in the table."""
        dataset_abbrs = [dataset_abbr_from_cfg(dataset) for dataset in self.dataset_cfgs]
        prompt_version = {dataset_abbr_from_cfg(d): get_prompt_hash(d)[:6] for d in self.dataset_cfgs}

        summarizer_dataset_abbrs = []
        if required_dataset_abbrs is None:
            # display all dataset metrics included in the config
            for dataset_abbr in dataset_abbrs:
                if dataset_abbr in dataset_metrics:
                    for metric in dataset_metrics[dataset_abbr]:
                        summarizer_dataset_abbrs.append((dataset_abbr, metric))
                else:
                    summarizer_dataset_abbrs.append((dataset_abbr, None))
            # along with all possible group metrics
            for dataset_abbr in dataset_metrics:
                for metric in dataset_metrics[dataset_abbr]:
                    if (dataset_abbr, metric) not in summarizer_dataset_abbrs:
                        summarizer_dataset_abbrs.append((dataset_abbr, metric))
        else:
            # follow the required order
            for item in required_dataset_abbrs:
                if isinstance(item, str):
                    summarizer_dataset_abbrs.append((item, None))
                elif isinstance(item, (list, tuple)):
                    summarizer_dataset_abbrs.append((item[0], item[1]))

        table = []
        header = ['dataset', 'version', 'metric', 'mode'] + self.model_abbrs
        table.append(header)
        
        for dataset_abbr, metric in summarizer_dataset_abbrs:
            if dataset_abbr not in dataset_metrics:
                if not skip_all_slash:
                    table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                continue
            if metric is None:
                metric = dataset_metrics[dataset_abbr][0]
            elif metric in dataset_metrics[dataset_abbr]:
                pass
            else:
                if not skip_all_slash:
                    table.append([dataset_abbr, '-', '-', '-'] + ['-'] * len(self.model_abbrs))
                continue

            row = [dataset_abbr, prompt_version.get(dataset_abbr, '-'), metric, dataset_eval_mode.get(dataset_abbr, '-')]
            for model_abbr in self.model_abbrs:
                if dataset_abbr in parsed_results[model_abbr]:
                    value = parsed_results[model_abbr][dataset_abbr].get(metric, '-')
                    if isinstance(value, float):
                        # Format API metrics with appropriate precision
                        if metric in ['avg_latency', 'min_latency', 'max_latency']:
                            row.append(f'{value:.3f}')
                        elif metric in ['total_tokens', 'total_requests', 'total_prompt_tokens', 'total_completion_tokens']:
                            row.append(f'{value:.0f}')
                        else:
                            row.append(f'{value:.2f}')
                    else:
                        row.append(str(value))
                else:
                    row.append('-')
            table.append(row)
        return table
