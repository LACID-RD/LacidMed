import matplotlib.pyplot as plt

def plot(H_x, H_y, rot_list):
    # Adding (1,0) as the starting point
    H_x = [0] + H_x
    H_y = [0] + H_y
    rot_list = [0] + rot_list

    # Define x values ensuring they are natural numbers and limited to 5
    x_values = list(range(1, min(len(H_x) + 1, len(H_x)+1)))

    # Create the figure
    plt.figure(figsize=(10, 5))

    # Plot translation in X
    plt.subplot(1, 3, 1)
    plt.plot(x_values, H_x[:len(x_values)], marker='o')
    plt.xticks(x_values)  # Ensure x-axis has only natural numbers
    plt.xlim(1, len(H_x))  # Set x-axis limits
    plt.title('Translation in X')
    plt.xlabel('Image Number')
    plt.ylabel('Translation Value [mm]')

    # Plot translation in Y
    plt.subplot(1, 3, 2)
    plt.plot(x_values, H_y[:len(x_values)], marker='o')
    plt.xticks(x_values)
    plt.xlim(1, len(H_x))
    plt.title('Translation in Y')
    plt.xlabel('Image Number')
    plt.ylabel('Translation Value [mm]')

    # Plot rotation
    plt.subplot(1, 3, 3)
    plt.plot(x_values, rot_list[:len(x_values)], marker='o')
    plt.xticks(x_values)
    plt.xlim(1, len(H_x))
    plt.title('Rotation')
    plt.xlabel('Image Number')
    plt.ylabel('Rotation Degree')

    plt.tight_layout()
    plt.show()
