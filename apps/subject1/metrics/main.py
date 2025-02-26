import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def count_by_direction_and_lane(data):
    return data.groupby(['Direction', 'Lane ID']).size().to_dict()

def accuracy_traffic_count_by_direction_by_lane(gt_count, infer_count):
    ape = (abs(gt_count - infer_count) / gt_count ) * 100
    return 100 - ape

def count_by_direction_and_lane_and_vehicle_type(data):
    return data.groupby(['Direction', 'Lane ID', 'Vehicle Type']).size().to_dict()

def accuracy_classification_by_direction_by_lane(gt_count, infer_count):
    pe = (abs(gt_count - infer_count) / gt_count ) * 100
    return 100 - pe

def count_queue_count_by_lane(data):
    data = data[data['In-Lane Time'].notna()]
    return data.groupby(['Lane ID']).size().to_dict()

def accuracy_queue_count_by_lane(gt_count, infer_count):
    pe = (abs(gt_count - infer_count) / gt_count ) * 100
    return 100 - pe

def plot_and_save(data, title, filename, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[x[0] for x in data], y=[x[1] for x in data], palette="coolwarm")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ha='right', wrap=True)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    gt_raw_data = pd.read_csv('groundtruth_directions.csv')
    infer_raw_data = pd.read_csv('inference_directions.csv')

    gt_counts = count_by_direction_and_lane(gt_raw_data)
    infer_counts = count_by_direction_and_lane(infer_raw_data)
    accuracy_traffic = [(f"{direction}\nLane {lane}", accuracy_traffic_count_by_direction_by_lane(gt_counts.get((direction, lane), 0), infer_counts.get((direction, lane), 0))) for (direction, lane) in gt_counts]
    plot_and_save(accuracy_traffic, "Traffic Count Accuracy by Direction and Lane", "traffic_count_accuracy.png", "Direction - Lane", "Accuracy (%)")

    gt_counts = count_by_direction_and_lane_and_vehicle_type(gt_raw_data)
    infer_counts = count_by_direction_and_lane_and_vehicle_type(infer_raw_data)
    accuracy_classification = [(f"{direction}\nLane {lane}\n{vehicle_type}", accuracy_classification_by_direction_by_lane(gt_counts.get((direction, lane, vehicle_type), 0), infer_counts.get((direction, lane, vehicle_type), 0))) for (direction, lane, vehicle_type) in gt_counts]
    plot_and_save(accuracy_classification, "Classification Accuracy by Direction, Lane, and Vehicle Type", "classification_accuracy.png", "Direction - Lane - Vehicle Type", "Accuracy (%)")

    gt_counts = count_queue_count_by_lane(gt_raw_data)
    infer_counts = count_queue_count_by_lane(infer_raw_data)
    accuracy_queue = [(f"Lane {lane}", accuracy_queue_count_by_lane(gt_counts.get(lane, 0), infer_counts.get(lane, 0))) for lane in gt_counts]
    plot_and_save(accuracy_queue, "Queue Count Accuracy by Lane", "queue_count_accuracy.png", "Lane", "Accuracy (%)")

if __name__ == "__main__":
    main()