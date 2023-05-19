import os
import pickle

current_directory = os.path.dirname(__file__)
pkl_dir = os.path.join(current_directory, 'path/to/your/results/')

def weekly_spotlight_inference_inhouse(year_week_list):
    """
    Load weekly spotlight inference results from pickle files.

    Args:
        year_week_list (list): List of strings containing year and week numbers, e.g., ['2022_8', '2022_9']

    Returns:
        final_result (list): List of dictionaries containing the inference results for each week.
    """
    final_result = []
    for y_w in year_week_list:
        try:
            overall_kw_df, output_di = pickle.load(open(pkl_dir + y_w + '.pkl', 'rb'))
            final_result.append(output_di)
        except FileNotFoundError:
            print(f"!!!Error: {y_w} hasn't been inferred yet and will be updated in the next version.")

    return final_result

if __name__ == '__main__':
    # Demo:
    # User Inputs: list of strings (combine year and week with an underscore), e.g., ['2022_8', '2022_9']
    # Returns a list of dictionaries
    user_input = ['2022_29']  # Example input

    print(list(weekly_spotlight_inference_inhouse(user_input)[0].keys()))
    # print(weekly_spotlight_inference_inhouse(user_input))