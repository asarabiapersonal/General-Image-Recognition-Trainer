import torch
import torch.optim as optim
import torch.nn as nn
import streamlit as st
import os
import numpy as np
import pandas as pd
import random
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install the canvas library: pip install streamlit-drawable-canvas")

# Import from our other files
import config
import utils
from model import Net
from dataloader import get_dataloaders, get_custom_dataloader_from_path, transform_test

# 1. Streamlit Configuration
st.set_page_config(page_title="Model Monitor", layout="wide")

# 2. Persistence Setup
if 'net' not in st.session_state:
    st.session_state['net'] = Net(num_classes=2).to(config.DEVICE)
    print("Model initialized in Session State.")

if 'optimizer' not in st.session_state:
    st.session_state['optimizer'] = optim.Adam(st.session_state['net'].parameters(), lr=config.LEARNING_RATE)

if 'last_loaded_path' not in st.session_state:
    st.session_state['last_loaded_path'] = None

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
if 'custom_val_path' not in st.session_state:
    st.session_state['custom_val_path'] = None

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
        train_dir=config.TRAIN_DIR,
        test_dir=config.TEST_DIR,
        batch_size=batch_size
    )

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
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

# --- DIGIT DRAWER DIALOG ---
@st.dialog("Write Digits / Draw Input")
def draw_digit_interface(class_names):
    st.caption("Draw on the canvas below to run inference with the selected model.")
    col_tools1, col_tools2 = st.columns(2)
    stroke_width = col_tools1.slider("Pen Width", 1, 30, 10)
    zoom_scale = col_tools2.select_slider("Zoom Level", options=[1, 2, 3, 4], value=2)
    canvas_size = 128 * zoom_scale
    
    if os.path.exists(config.MODEL_SAVE_DIR):
        model_files = [f for f in os.listdir(config.MODEL_SAVE_DIR) if f.endswith('.pth')]
        model_files.sort(reverse=True)
        current_idx = 0
        if st.session_state['current_loaded_filename'] in model_files:
            current_idx = model_files.index(st.session_state['current_loaded_filename'])
        selected_model = st.selectbox("Select Model for Inference", model_files, index=current_idx)
        
        if selected_model != st.session_state['current_loaded_filename']:
            full_path = os.path.join(config.MODEL_SAVE_DIR, selected_model)
            utils.load_specific_model(full_path)
            st.session_state['current_loaded_filename'] = selected_model
            st.session_state['last_loaded_path'] = full_path
            
    st.divider()

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict Drawing", type="primary"):
        if canvas_result.image_data is not None:
            img_data = canvas_result.image_data
            image = Image.fromarray(img_data.astype('uint8'), mode="RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3]) 
            image = background
            image = image.resize((128, 128))
            input_tensor = transform_test(image)
            input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
            
            net = st.session_state['net']
            net.eval()
            with torch.no_grad():
                output = net(input_tensor)
                probs = torch.exp(output)
                confidences, predicted = torch.max(probs, 1)
                
            pred_idx = predicted.item()
            conf_score = confidences.item() * 100
            
            if pred_idx < len(class_names):
                pred_label = class_names[pred_idx]
            else:
                pred_label = f"Unknown Class ID {pred_idx}"
            
            st.success(f"Prediction: **{pred_label}**")
            st.info(f"Confidence: **{conf_score:.2f}%**")
            st.image(image, caption="Resized Input (128x128)", width=128)
            
            # Show Feature Maps
            utils.visualize_feature_maps(net, input_tensor)

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
            
            # --- NEW: Feature Maps Expander ---
            with st.expander(f"👁️ View Feature Maps"):
                # Get the single image tensor back [1, 3, 128, 128]
                single_input = images[i].unsqueeze(0).to(config.DEVICE)
                utils.visualize_feature_maps(net, single_input)

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
        running_loss = 0.0
        
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
            
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            plot_every = 10 
            if (i + 1) % plot_every == 0:
                avg_loss = running_loss / plot_every
                final_avg_loss = avg_loss
                current_step = i + 1 + (epoch * len(trainloader))
                
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
                
                running_loss = 0.0
    
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
    
    current_model_path = st.session_state.get('last_loaded_path')
    if current_model_path:
        model_name = os.path.basename(current_model_path)
        st.write(f"**📂 Currently Loaded Model:** `{model_name}`")
    else:
        st.write("**📂 Currently Loaded Model:** `None (New/Untrained)`")
    
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

    st.sidebar.header("Hyperparameters")
    hp_batch_size = st.sidebar.slider("Batch Size", 16, 128, config.BATCH_SIZE, 16)
    hp_epochs = st.sidebar.slider("Epochs", 1, 20, config.EPOCHS)
    hp_lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, config.LEARNING_RATE, format="%.4f", step=0.0005)
    
    trainloader, testloader, class_names = load_default_data(hp_batch_size)
    
    if st.session_state.get('custom_train_path'):
        try:
            custom_train_path = st.session_state['custom_train_path']
            custom_train_loader, custom_train_classes = get_custom_dataloader_from_path(
                custom_train_path, batch_size=hp_batch_size, is_train=True
            )
            trainloader = custom_train_loader
            class_names = custom_train_classes 
        except Exception as e:
            st.error(f"Error loading custom training data: {e}")

    if st.session_state.get('custom_val_path'):
        try:
            custom_val_path = st.session_state['custom_val_path']
            custom_val_loader, _ = get_custom_dataloader_from_path(
                custom_val_path, batch_size=hp_batch_size, is_train=False
            )
            testloader = custom_val_loader
        except Exception as e:
            st.error(f"Error loading custom validation data: {e}")

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
    if trackers['cm_true'] and trackers['cm_pred'] and int(model_output_features) < 12:
        st.markdown("### 📊 Confusion Matrix")
        fig_cm = utils.create_confusion_matrix_chart(trackers['cm_true'], trackers['cm_pred'], class_names)
        if fig_cm:
            st.pyplot(fig_cm, use_container_width=True)
    elif model_output_features >= 12:
        st.info("Confusion Matrix display is disabled for more than 10 classes.")

    st.sidebar.header("Execution Controls")
    col_check1, col_check2 = st.sidebar.columns(2)
    do_train = col_check1.checkbox("Train Model", value=True)
    if do_train:
        if col_check1.button("Select Train Folder"):
            folder = select_folder()
            if folder:
                st.session_state['custom_train_path'] = folder
                st.toast(f"Train Source: {folder}")
                st.rerun()
        if st.session_state.get('custom_train_path'):
            col_check1.caption(f"`{st.session_state['custom_train_path']}`")

    can_validate = st.session_state['model_is_ready'] or do_train
    do_validate = col_check2.checkbox("Validate Model", value=False, disabled=not can_validate)
    if do_validate:
        if col_check2.button("Select Valid Folder"):
            folder = select_folder()
            if folder:
                st.session_state['custom_val_path'] = folder
                st.toast(f"Val Source: {folder}")
                st.rerun()
        if st.session_state.get('custom_val_path'):
            col_check2.caption(f"`{st.session_state['custom_val_path']}`")

    new_model_name = st.sidebar.text_input("Name for New Model (Optional)", placeholder="e.g. my_best_model")

    if st.sidebar.button("Run Execution"):
        
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
                session_metrics,
                custom_name=new_model_name 
            )
            if do_validate:
                st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.header("Model Management")
    
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)
        
    model_files = [f for f in os.listdir(config.MODEL_SAVE_DIR) if f.endswith('.pth')]
    model_files.sort(reverse=True)
    
    selected_file_name = st.sidebar.selectbox("Select Model File", model_files)
    
    if selected_file_name and selected_file_name != st.session_state['current_loaded_filename']:
        full_path = os.path.join(config.MODEL_SAVE_DIR, selected_file_name)
        utils.load_specific_model(full_path)
        st.session_state['current_loaded_filename'] = selected_file_name
        st.session_state['last_loaded_path'] = full_path
        st.rerun()

    if selected_file_name:
        new_name_input = st.sidebar.text_input("Rename Current Model", value=selected_file_name.replace('.pth', ''))
        if st.sidebar.button("Rename"):
            new_path, err = utils.rename_model(selected_file_name, new_name_input)
            if new_path:
                st.success(f"Renamed to {os.path.basename(new_path)}")
                st.session_state['current_loaded_filename'] = os.path.basename(new_path)
                st.session_state['last_loaded_path'] = new_path
                st.rerun()
            else:
                st.error(f"Rename failed: {err}")

    if st.sidebar.toggle("Show Drawing Canvas"):
        draw_digit_interface(class_names)

    col_test1, col_test2 = st.sidebar.columns(2)
    
    if col_test1.button("Select Test Folder"):
        folder = select_folder()
        if folder:
            st.session_state['custom_test_path'] = folder
            st.toast(f"Test Source: {folder}")
            st.rerun()

    if col_test2.button("Test Batch"):
        active_test_loader = testloader 
        active_test_classes = class_names
        
        if st.session_state.get('custom_test_path'):
             try:
                custom_path = st.session_state['custom_test_path']
                t_loader, t_classes = get_custom_dataloader_from_path(
                    custom_path, batch_size=16, is_train=False
                )
                active_test_loader = t_loader
                active_test_classes = t_classes
             except Exception as e:
                 st.error(f"Error loading custom test data: {e}")

        run_inference(active_test_loader, active_test_classes)
    
    if st.session_state.get('custom_test_path'):
        st.sidebar.caption(f"Test: `{st.session_state['custom_test_path']}`")

    cpu, ram, gpu_info, gpu_mem = utils.get_system_metrics()
    metric_cpu.metric("CPU", f"{cpu}%")
    metric_ram.metric("RAM", f"{ram}%")
    metric_gpu_name.caption(f"**GPU:** {gpu_info}")
    metric_gpu_mem.metric("VRAM", gpu_mem)

if __name__ == '__main__':
    main()