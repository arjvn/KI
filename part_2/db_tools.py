# def parse_class_labels(filename):
#     class_id_to_name = {}
#     with open(filename, 'r') as file:
#         for line in file:
#             parts = line.strip().split('\t')
#             class_id = parts[0]
#             class_name = parts[1].split(',')[0]  # Taking the first name if there are multiple names
#             class_id_to_name[class_id] = class_name
#     return class_id_to_name


# def parse_ids_to_int(filename):
#     label_to_index = {}
#     with open(filename, 'r') as file:
#         for index, line in enumerate(file, start=1):  # start=1 to start numbering from 1
#             parts = line.strip().split('\t')
#             class_id = parts[0]
#             label_to_index[class_id] = index
#     return label_to_index

# def parse_int_to_label(filename):
#     index_to_description = {}  # Create an empty dictionary
    
#     with open(filename, 'r') as file:
#         for index, line in enumerate(file, start=1):  # Start indexing from 1
#             parts = line.strip().split('\t')  # Split each line by the tab character
#             if len(parts) < 2:
#                 continue  # If the line doesn't have at least two parts, skip it
#             description = parts[1]  # The second part is the description
#             index_to_description[str(index)] = description  # Map index as string to description
    
#     return index_to_description

def id_to_name(filename):
    class_id_to_name = {}  # Create an empty dictionary to store the mappings

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')  # Split the line on the tab character
            if len(parts) == 2:  # Ensure there are exactly two parts
                class_id = parts[0]  # The class ID is the first part
                description = parts[1]  # The description is the second part
                class_id_to_name[class_id] = description  # Map the class ID to the description

    return class_id_to_name
