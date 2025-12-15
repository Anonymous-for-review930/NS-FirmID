import json
import os.path


def add_new_data(new_data_path, output_path):
    """
    Add new data to the database

    Args:
        new_data_path: Path to the new data file
        output_path: Path to save the updated database

    Returns:
        None
    """
    # Note: You might want to ensure this file exists or handle the file not found error
    with open('firmware_database_all_brand_0913_clean.json', 'r', encoding='utf-8') as f:
        data_base = json.load(f)
        init_base_len = len(list(data_base.keys()))
        print(f"Initial database length: {init_base_len}")

        with open(new_data_path, 'r', encoding='utf-8') as f:
            try:
                # Try loading as a standard JSON object
                new_data = json.load(f)
            except json.decoder.JSONDecodeError:
                # If that fails, try loading as JSON Lines (one JSON object per line)
                with open(new_data_path, 'r', encoding='utf-8') as f:
                    new_data = []
                    for line in f:
                        new_data.append(json.loads(line))

            # Handle case where new_data is a dictionary (nested structure)
            if isinstance(new_data, dict):
                for brand, data in new_data.items():
                    if brand in data_base:
                        for model, version_list in data:
                            if model in data_base[brand].keys():
                                for version in version_list:
                                    if version not in data_base[brand][model]:
                                        data_base[brand][model].append(version)
                            else:
                                data_base[brand][model] = version_list
                    else:
                        data_base[brand] = data

            # Handle case where new_data is a list (flat structure)
            if isinstance(new_data, list):
                # Assume filename represents the brand
                brand = os.path.basename(new_data_path).split('.json')[0]
                for data in new_data:
                    model = data["model"]
                    version = data["version"]
                    if brand in data_base.keys():
                        if model in data_base[brand].keys():
                            if version not in data_base[brand][model]:
                                data_base[brand][model].append(version)
                        else:
                            data_base[brand][model] = [version]
                    else:
                        data_base[brand] = {model: [version]}

        new_len = len(list(data_base.keys()))

        # Save the updated data if changes occurred (or simply overwrite)
        if new_len >= init_base_len:
            # Save brand-centric database
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(data_base, f_out, ensure_ascii=False, indent=4)

            # Generate and save model-centric database
            model_path = output_path.replace("brand", "model")
            model_data = {}
            for brand, models in data_base.items():
                for model, versions in models.items():
                    if model not in model_data:
                        model_data[model] = {}
                    if brand not in model_data[model]:
                        model_data[model][brand] = versions
                    else:
                        for version in versions:
                            if version not in model_data[model][brand]:
                                model_data[model][brand].append(version)

            with open(model_path, 'w', encoding='utf-8') as f_out:
                json.dump(model_data, f_out, ensure_ascii=False, indent=4)

            print(f"✓ Database updated! Added {new_len - init_base_len} new records.")


if __name__ == "__main__":
    new_path = "F:\\paper\\paper_gu\\firmware_version_identification\\spider_module\\myfirstSpider\\result_modified_new2\\innovaphone.json"
    add_new_data(new_path, "firmware_database_all_brand_1118.json")
