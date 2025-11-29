# train.py

import torch
import torch.optim as optim
import torch.nn as nn
<<<<<<< HEAD
import streamlit as st
import os
import numpy as np
import pandas as pd
import random
import time
import tkinter as tk
from tkinter import filedialog
=======
import matplotlib.pyplot as plt
>>>>>>> parent of 43cbc7e (restructure)

# Import from our other files
import config
from model import Net
from dataloader import get_dataloaders, get_custom_dataloader_from_path

<<<<<<< HEAD
# 1. Streamlit Configuration
st.set_page_config(page_title="Model Monitor", layout="wide")

# 2. Persistence Setup
if 'net' not in st.session_state:
    st.session_state['net'] = Net(num_classes=2).to(config.DEVICE)
    print("Model initialized in Session State.")
=======
stepTracker = []
batchTracker = []
val_step = []
val_accuracy = []

>>>>>>> parent of 43cbc7e (restructure)


<<<<<<< HEAD
if 'last_loaded_path' not in st.session_state:
    st.session_state['last_loaded_path'] = None

# New state to track the filename specifically for the dropdown logic
if 'current_loaded_filename' not in st.session_state:
    st.session_state['current_loaded_filename'] = None

if 'model_is_ready' not in st.session_state:
    st.session_state['model_is_ready'] = False

if 'active_model_meta' not in st.session_state:
    st.session_state['active_model_meta'] = {}

# Custom Paths
if 'custom_test_path' not in st.session_state:
    st.session_state['custom_test_path'] = None
if 'custom_train_path' not in st.session_state:
    st.session_state['custom_train_path'] = None

if 'trackers' not in st.session_state:
    st.session_state['trackers'] = {
        'stepTracker': [],
        'batchTracker': [],
        'val_step': [],
        'val_accuracy': [],
        'val_batch_idx': 0,
        'cm_true': [],
        'cm_pred': []
    }

if 'log_history' not in st.session_state:
    st.session_state['log_history'] = []

# 3. Load Data
@st.cache_resource
def load_default_data(batch_size):
    return get_dataloaders(
=======
def main():
    """Main function to run the training and validation."""
    torch.manual_seed(config.MANUAL_SEED)
    
    # 1. Load Data
    trainloader, testloader, class_names = get_dataloaders(
>>>>>>> parent of 43cbc7e (restructure)
        train_dir=config.TRAIN_DIR,
        test_dir=config.TEST_DIR,
        batch_size=batch_size
    )
    print("Data loaded successfully.")
    print(f"Classes: {class_names}")

<<<<<<< HEAD
criterion = nn.NLLLoss()

# --- HELPER: TERMINAL LOGGING ---
def log_message(message, terminal_placeholder):
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    st.session_state['log_history'].append(formatted_msg)
    if len(st.session_state['log_history']) > 20:
        st.session_state['log_history'].pop(0)
    log_text = "\n".join(st.session_state['log_history'])
    if terminal_placeholder:
        terminal_placeholder.code(log_text, language="bash")

# --- UI HELPER: FOLDER SELECTION ---
def select_folder():
    """Opens a local system dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

# --- EXECUTION FUNCTIONS ---

def run_inference(default_testloader, default_class_names):
    if not st.session_state['model_is_ready']:
        st.error("Model is not trained or loaded.")
        return

    st.subheader("🖼️ Live Inference Test")
    
    active_loader = default_testloader
    active_class_names = default_class_names
    source_msg = "Using Default Test Set"

    custom_path = st.session_state.get('custom_test_path')
    
    if custom_path:
        try:
            custom_loader, custom_classes = get_custom_dataloader_from_path(custom_path, batch_size=16, is_train=False)
            active_loader = custom_loader
            active_class_names = custom_classes
            source_msg = f"Using Custom Folder: `{os.path.basename(custom_path)}`"
        except Exception as e:
            st.error(f"Failed to load custom images from {custom_path}. Falling back to default. Error: {e}")

    st.caption(source_msg)

    net = st.session_state['net']
    net.eval()
    
    if len(active_loader.dataset) == 0:
        st.error("Dataset is empty.")
        return

    try:
        dataset = active_loader.dataset
        num_to_show = 4
        if len(dataset) < num_to_show:
            num_to_show = len(dataset)
            
        indices = random.sample(range(len(dataset)), num_to_show)
        batch_images = []
        batch_labels = []
        for idx in indices:
            image, label = dataset[idx]
            batch_images.append(image)
            batch_labels.append(label)
        images = torch.stack(batch_images)
        labels = torch.tensor(batch_labels)
    except:
        dataiter = iter(active_loader)
        images, labels = next(dataiter)
        num_to_show = min(4, images.size(0))
        images = images[:num_to_show]
        labels = labels[:num_to_show]

    images_device = images.to(config.DEVICE)
    
    with torch.no_grad():
        outputs = net(images_device)
        probs = torch.exp(outputs)
        confidences, predicted = torch.max(probs, 1)
        
    cols = st.columns(num_to_show)
    
    for i in range(num_to_show):
        with cols[i]:
            img_tensor = images[i] * 0.5 + 0.5
            np_img = img_tensor.numpy()
            np_img = np.transpose(np_img, (1, 2, 0))
            
            pred_idx = predicted[i].item()
            true_idx = labels[i].item()
            
            if true_idx < len(active_class_names):
                true_label = active_class_names[true_idx]
            else:
                true_label = f"Unknown ({true_idx})"
                
            if pred_idx < len(active_class_names):
                pred_label = active_class_names[pred_idx]
            else:
                pred_label = f"Unknown ({pred_idx})"
            
            confidence_pct = confidences[i].item() * 100
            
            if true_idx == pred_idx:
                header_color = "green"
                result_text = "✅ Correct"
            else:
                header_color = "red"
                result_text = "❌ Incorrect"
                
            st.image(np_img, use_container_width=True)
            st.markdown(f"**Actual:** {true_label}")
            st.markdown(f"**Pred:** :{header_color}[{pred_label}] ({confidence_pct:.1f}%)")
            st.caption(result_text)

def train_model(trainloader, loss_chart, metric_placeholders, terminal_placeholder, header_placeholder, epochs, lr, batch_size):
    net = st.session_state['net']
    st.session_state['optimizer'] = optim.Adam(net.parameters(), lr=lr)
    optimizer = st.session_state['optimizer']
    
    p_cpu, p_ram, p_gpu_name, p_gpu_mem = metric_placeholders
    
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    final_avg_loss = 0.0

    trackers = st.session_state['trackers']
    
    log_message("--- Training Started ---", terminal_placeholder)

    for epoch in range(epochs): 
        net.train()
=======
    # 2. Initialize Model, Optimizer, Loss
    net = Net().to(config.DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.NLLLoss()

    

    print("Model initialized.")
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        net.train() # Set model to training mode
>>>>>>> parent of 43cbc7e (restructure)
        running_loss = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
<<<<<<< HEAD
            
            plot_every = 10 
=======
            plot_every = 100 # Plot loss every 100 batches
>>>>>>> parent of 43cbc7e (restructure)
            if (i + 1) % plot_every == 0:
                avg_loss = running_loss / plot_every
                final_avg_loss = avg_loss
                current_step = i + 1 + (epoch * len(trainloader))
<<<<<<< HEAD
                
                trackers['stepTracker'].append(current_step)
                trackers['batchTracker'].append(avg_loss)
                
                new_data = pd.DataFrame({'Loss': [avg_loss]}, index=[current_step])
                loss_chart.add_rows(new_data)
                
                cpu, ram, gpu_info, gpu_mem = utils.get_system_metrics()
                p_cpu.metric("CPU", f"{cpu}%")
                p_ram.metric("RAM", f"{ram}%")
                p_gpu_name.caption(f"**GPU:** {gpu_info}")
                p_gpu_mem.metric("VRAM", gpu_mem)
                
                msg = f"Epoch {epoch+1}/{epochs} | Batch {i+1} | Loss: {avg_loss:.4f}"
                log_message(msg, terminal_placeholder)
                header_placeholder.markdown(f"## 🟢 Training: {msg}")
                print(msg) 
                
=======
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')

                # Append new data points for the loss plot
                stepTracker.append(current_step)
                batchTracker.append(avg_loss)
>>>>>>> parent of 43cbc7e (restructure)
                running_loss = 0.0

        print(f'Finished training epoch: {epoch + 1}')

        # --- VALIDATION ---
        net.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate overall accuracy for the epoch
        overall_accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {overall_accuracy:.2f} %')

        # Append data for the accuracy plot
        # We use the last step of the epoch for the x-axis value
        val_step.append(len(trainloader) * (epoch + 1))
        val_accuracy.append(overall_accuracy)

        # --- UPDATE PLOT INTERACTIVELY ---
        # Update data for both lines
        line.set_xdata(stepTracker)
        line.set_ydata(batchTracker)
        line2.set_xdata(val_step)
        line2.set_ydata(val_accuracy)

        # Rescale axes
        axis.relim()
        axis.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
    
<<<<<<< HEAD
    log_message("--- Training Finished ---", terminal_placeholder)
    return final_avg_loss

def validate_model(testloader, acc_chart, live_val_chart, terminal_placeholder, header_placeholder, class_names):
    net = st.session_state['net']
    net.eval()
    
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    trackers = st.session_state['trackers']
    
    log_message("--- Validation Started ---", terminal_placeholder)
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_acc = 100 * correct / total
            trackers['val_batch_idx'] += 1
            
            live_val_chart.add_rows(pd.DataFrame({'Running Acc': [current_acc]}, index=[i]))
            
            trackers['val_step'].append(trackers['val_batch_idx'])
            trackers['val_accuracy'].append(current_acc)
            acc_chart.add_rows(pd.DataFrame({'Accuracy': [current_acc]}, index=[trackers['val_batch_idx']]))
            
            if (i+1) % 5 == 0:
                msg = f"Batch {i+1} | Acc: {current_acc:.2f}%"
                log_message(msg, terminal_placeholder)
                header_placeholder.markdown(f"## 🟠 Validating: {msg}")

    accuracy = 0.0
    if total > 0:
        accuracy = 100 * correct / total
        log_message(f"--- Validation Complete. Final Acc: {accuracy:.2f}% ---", terminal_placeholder)
        st.info(f"Final Validation Accuracy: {accuracy:.2f}%")
        
        trackers['cm_true'] = all_labels
        trackers['cm_pred'] = all_preds
    else:
        st.warning("Validation set is empty.")
        
    return accuracy

def main():
    header_placeholder = st.empty()
    header_placeholder.title("🛡️ Model Performance & Health Monitor")
    
    # --- SIDEBAR: SYSTEM & INFO ---
    st.sidebar.markdown("### 🖥️ System Monitor")
    m_col1, m_col2 = st.sidebar.columns(2)
    metric_cpu = m_col1.empty()
    metric_ram = m_col2.empty()
    metric_gpu_mem = st.sidebar.empty()
    metric_gpu_name = st.sidebar.empty()
    st.sidebar.markdown("---")
    
    utils.display_active_model_info() 

    with st.sidebar.expander("🏗️ View Network Architecture"):
        input_size = (1, 3, 128, 128) 
        summary_txt = utils.get_model_summary(st.session_state['net'], input_size)
        st.code(summary_txt, language="text")

    # --- SIDEBAR: HYPERPARAMETERS ---
    st.sidebar.header("Hyperparameters")
    hp_batch_size = st.sidebar.slider("Batch Size", 16, 128, config.BATCH_SIZE, 16)
    hp_epochs = st.sidebar.slider("Epochs", 1, 20, config.EPOCHS)
    hp_lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, config.LEARNING_RATE, format="%.4f", step=0.0005)
    
    # --- LOAD DATA LOGIC ---
    # Default Load
    trainloader, testloader, class_names = load_default_data(hp_batch_size)
    
    # Check for Custom Train Path override
    if st.session_state.get('custom_train_path'):
        try:
            custom_train_path = st.session_state['custom_train_path']
            # Using is_train=True to get augmentations and shuffle
            custom_train_loader, custom_train_classes = get_custom_dataloader_from_path(
                custom_train_path, batch_size=hp_batch_size, is_train=True
            )
            # Override
            trainloader = custom_train_loader
            class_names = custom_train_classes # Classes defined by training folder
            # st.sidebar.info(f"Train: {os.path.basename(custom_train_path)}")
        except Exception as e:
            st.error(f"Error loading custom training data: {e}. Reverting to default.")

    # Check for Custom Test Path override
    if st.session_state.get('custom_test_path'):
        try:
            custom_test_path = st.session_state['custom_test_path']
            # Using is_train=False for test data
            custom_test_loader, _ = get_custom_dataloader_from_path(
                custom_test_path, batch_size=hp_batch_size, is_train=False
            )
            # Override
            testloader = custom_test_loader
            # st.sidebar.info(f"Test: {os.path.basename(custom_test_path)}")
        except Exception as e:
            st.error(f"Error loading custom test data: {e}. Reverting to default.")

    # --- SAFETY CHECK: CLASS MISMATCH ---
    current_num_classes = len(class_names)
    model_output_features = st.session_state['net'].fc3.out_features
    if model_output_features != current_num_classes:
        st.warning(f"⚠️ Class Count Mismatch! Resetting model...")
        st.session_state['net'] = Net(num_classes=current_num_classes).to(config.DEVICE)
        st.session_state['optimizer'] = optim.Adam(st.session_state['net'].parameters(), lr=hp_lr)
        st.session_state['model_is_ready'] = False
        st.session_state['last_loaded_path'] = None
        st.rerun() 
    
    trackers = st.session_state['trackers']

    # --- MAIN PAGE: CHARTS ---
    st.markdown("### 📈 Training & Validation Charts")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Training Loss**")
        loss_df = pd.DataFrame()
        if trackers['stepTracker'] and trackers['batchTracker']:
            loss_df = pd.DataFrame({'Loss': trackers['batchTracker']}, index=trackers['stepTracker'])
        loss_chart = st.line_chart(loss_df)

    with col_chart2:
        st.markdown("**Validation Accuracy**")
        acc_df = pd.DataFrame()
        if trackers['val_step'] and trackers['val_accuracy']:
            acc_df = pd.DataFrame({'Accuracy': trackers['val_accuracy']}, index=trackers['val_step'])
        acc_chart = st.line_chart(acc_df)
    
    st.markdown("**Live Validation Trend**")
    live_val_chart = st.line_chart(pd.DataFrame(columns=['Running Acc']))

    st.markdown("### 📟 Execution Log")
    terminal_placeholder = st.empty()
    if st.session_state['log_history']:
        terminal_placeholder.code("\n".join(st.session_state['log_history']), language="bash")
    else:
        terminal_placeholder.code("Ready...", language="bash")

    model_output_features = st.session_state['net'].fc3.out_features
    if trackers['cm_true'] and trackers['cm_pred'] and int(model_output_features) < 10:
        st.markdown("### 📊 Confusion Matrix")
        fig_cm = utils.create_confusion_matrix_chart(trackers['cm_true'], trackers['cm_pred'], class_names)
        if fig_cm:
            st.pyplot(fig_cm, use_container_width=True)
    elif model_output_features >= 10:
        st.info("Confusion Matrix display is disabled for more than 10 classes.")

    # --- SIDEBAR: EXECUTION CONTROLS ---
    st.sidebar.header("Execution Controls")
    
    # 1. Mode Selection
    col_check1, col_check2 = st.sidebar.columns(2)
    do_train = col_check1.checkbox("Train Model", value=True)
    can_validate = st.session_state['model_is_ready'] or do_train
    do_validate = col_check2.checkbox("Validate Model", value=False, disabled=not can_validate)

    # 2. Run Buttons
    col_run1, col_run2 = st.sidebar.columns(2)
    
    # BUTTON: Run Execution
    if col_run1.button("Run Execution"):
        session_metrics = {}
        if not do_train:
             session_metrics = st.session_state['active_model_meta'].get('metrics', {}).copy()
        
        if do_train:
            trackers['stepTracker'].clear()
            trackers['batchTracker'].clear()
            loss_chart.add_rows(pd.DataFrame(columns=['Loss']))
            st.session_state['log_history'] = [] 
            terminal_placeholder.code("Starting...", language="bash")
            
        if do_validate:
            trackers['val_step'].clear()
            trackers['val_accuracy'].clear()
            trackers['val_batch_idx'] = 0
            trackers['cm_true'].clear()
            trackers['cm_pred'].clear()
            acc_chart.add_rows(pd.DataFrame(columns=['Accuracy']))
            live_val_chart.add_rows(pd.DataFrame(columns=['Running Acc']))
        
        if do_train:
            loss = train_model(
                trainloader, 
                loss_chart, 
                [metric_cpu, metric_ram, metric_gpu_name, metric_gpu_mem], 
                terminal_placeholder,
                header_placeholder, 
                hp_epochs, 
                hp_lr, 
                hp_batch_size
            )
            session_metrics['final_loss'] = loss
            st.session_state['model_is_ready'] = True
            
        if do_validate:
            if st.session_state['model_is_ready']:
                accuracy = validate_model(testloader, acc_chart, live_val_chart, terminal_placeholder, header_placeholder, class_names)
                session_metrics['val_accuracy'] = accuracy
            else:
                st.error("Cannot validate: Model not ready.")

        header_placeholder.title("🛡️ Model Performance & Health Monitor")

        if do_train or do_validate:
            utils.save_checkpoint(
                st.session_state['net'],
                st.session_state['optimizer'],
                hp_epochs,
                hp_lr,
                hp_batch_size,
                session_metrics
            )
            if do_validate:
                st.rerun()

    # BUTTON: Select Train Folder (Moved here as requested)
    if col_run2.button("Select Train Folder"):
        folder = select_folder()
        if folder:
            st.session_state['custom_train_path'] = folder
            st.toast(f"Train Source: {os.path.basename(folder)}")
            st.rerun()

    # Show active train path
    if st.session_state.get('custom_train_path'):
        st.sidebar.caption(f"Train: `{os.path.basename(st.session_state['custom_train_path'])}`")

    st.sidebar.markdown("---")

    # --- SIDEBAR: MODEL MANAGEMENT ---
    st.sidebar.header("Model Management")
    
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)
        
    model_files = [f for f in os.listdir(config.MODEL_SAVE_DIR) if f.endswith('.pth')]
    model_files.sort(reverse=True) # Sort newest first usually better
    
    # AUTO-LOAD LOGIC for Dropdown
    selected_file_name = st.sidebar.selectbox("Select Model File", model_files)
    
    # Track current loaded file to detect changes
    if selected_file_name and selected_file_name != st.session_state['current_loaded_filename']:
        full_path = os.path.join(config.MODEL_SAVE_DIR, selected_file_name)
        utils.load_specific_model(full_path)
        # Update state so we don't reload on next frame
        st.session_state['current_loaded_filename'] = selected_file_name
        st.session_state['last_loaded_path'] = full_path
        st.rerun()

    # --- TEST BUTTONS (Moved under Model Selection) ---
    col_test1, col_test2 = st.sidebar.columns(2)
    
    if col_test1.button("Select Test Folder"):
        folder = select_folder()
        if folder:
            st.session_state['custom_test_path'] = folder
            st.toast(f"Test Source: {os.path.basename(folder)}")
            st.rerun()

    if col_test2.button("Test Batch"):
        run_inference(testloader, class_names)
    
    # Show active test path
    if st.session_state.get('custom_test_path'):
        st.sidebar.caption(f"Test: `{os.path.basename(st.session_state['custom_test_path'])}`")

    # Initial metric update
    cpu, ram, gpu_info, gpu_mem = utils.get_system_metrics()
    metric_cpu.metric("CPU", f"{cpu}%")
    metric_ram.metric("RAM", f"{ram}%")
    metric_gpu_name.caption(f"**GPU:** {gpu_info}")
    metric_gpu_mem.metric("VRAM", gpu_mem)
=======
    print("Finished training.")

    # 4. Save the model
    torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

>>>>>>> parent of 43cbc7e (restructure)

if __name__ == '__main__':
    main()