import os
import pandas as pd
import numpy as np


class ServeDataProcessor:
    def __init__(self, csv_file, output_directory, standard_file):
        self.csv_file = csv_file
        self.output_directory = output_directory
        self.standard_file = standard_file
        self.column_names = [   "Frame", "Serve Count", "Wrist to Hip Distance (pixels)", "Shoulder Angle (degrees)",
            "Trunk Rotation (degrees)", "Wrist Flexion (degrees)", "Impact Height (pixels)",
            "Step Distance (pixels)", "Knee Flexion (degrees)"]

    def process_file(self):
        if not os.path.exists(self.csv_file):
            print(f"File {self.csv_file} does not exist.")
            return

        df = pd.read_csv(self.csv_file)

        # Filter out serve count 0
        df = df[df['Serve Count'] != 0]

        # Process data for each serve (only for launch phase)
        serve_stats = self.aggregate_data_by_serve(df)

        # Compare results within the file
        ranges = self.compare_data(serve_stats)

        # Compare with standard results
        classification, score = self.compare_with_standard(ranges)

        # Output the aggregated and comparison results
        self.save_results(ranges, classification, score)

    def aggregate_data_by_serve(self, df):
        serve_data = {}

        for serve_count in df['Serve Count'].unique():
            if serve_count == 0:
                continue

            serve_df = df[df['Serve Count'] == serve_count]
            total_frames = len(serve_df)

            first_section_end = int(total_frames * 0.35)
            second_section_end = int(total_frames * 0.60)

            power_generation_section = serve_df.iloc[:first_section_end // 2]
            impact_section = serve_df.iloc[first_section_end:second_section_end]

            serve_data[serve_count] = {
                'power_generation': power_generation_section.mean().to_dict(),
                'impact': impact_section.mean().to_dict()
            }

        return serve_data

    def compare_data(self, serve_data):
        ranges = {}

        for param in self.column_names[2:]:
            param_values = []

            for serve_count, serve_info in serve_data.items():
                if 'power_generation' in serve_info:
                    param_values.append(serve_info['power_generation'].get(param, np.nan))
                if 'impact' in serve_info:
                    param_values.append(serve_info['impact'].get(param, np.nan))

            param_values = np.array(param_values)

            if len(param_values) == 0:
                continue

            Q1 = np.percentile(param_values, 25)
            Q3 = np.percentile(param_values, 75)
            IQR = Q3 - Q1

            filtered_values = param_values[(param_values >= (Q1 - 1.5 * IQR)) & (param_values <= (Q3 + 1.5 * IQR))]

            ranges[param] = {
                'min': np.min(filtered_values),
                'max': np.max(filtered_values)
            }

        return ranges

    def compare_with_standard(self, ranges):
        standard_df = pd.read_csv(self.standard_file, index_col=0)
        match_count = 0
        total_params = len(self.column_names[2:])

        for param in self.column_names[2:]:
            if param in ranges and param in standard_df.index:
                min_standard, max_standard = standard_df.loc[param, 'min'], standard_df.loc[param, 'max']
                min_ours, max_ours = ranges[param]['min'], ranges[param]['max']

                overlap = max(0, min(max_ours, max_standard) - max(min_ours, min_standard))
                total_range = max_standard - min_standard

                if total_range > 0 and (overlap / total_range) > 0.4:
                    match_count += 1

        if 2 <= match_count <= 4:
            classification = "Intermediate"
            score = 4 + (match_count - 4)
        else:
            classification = "Beginner"
            score = 3

        if match_count >= 5:
            classification = "Advanced"
            score = 7 + (match_count - 4)

        return classification, score

    def save_results(self, ranges, classification, score):
        output_file = os.path.join(self.output_directory, 'parameter_ranges_power.csv')
        range_data = pd.DataFrame.from_dict(ranges, orient='index')
        range_data.to_csv(output_file)
        print(f'Results saved to {output_file}')
        print(f'Player Classification: {classification}, Score: {score}')
