import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import streamlit as st
import os
import glob
import re
from torchinfo import summary
import seaborn as sns
from sklearn.metrics import confusion_matrix
import config
from model import Net
import torch.optim as optim
import gc
from PIL import Image  # <--- Added Import for resizing

# --- CHARTING HELPERS ---
def create_confusion_matrix_chart(y_true, y_pred, class_names):
    if not y_true or not y_pred:
        return None
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 4)) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        return fig
    except Exception as e:
        return None

def visualize_feature_maps(model, input_tensor, layer_names=['conv1', 'conv2', 'conv3']):
    """
    Passes an image through the model and visualizes the feature maps.
    UPSCALES feature maps to 128x128 for clear viewing.
    """
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    for name, layer in model.named_children():
        if name in layer_names:
            h = layer.register_forward_hook(get_activation(name))
            hooks.append(h)

    # Forward pass
    model.eval()
    with torch.no_grad():
        if input_tensor.device != config.DEVICE:
            input_tensor = input_tensor.to(config.DEVICE)
        model(input_tensor)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Visualization
    cmap = plt.get_cmap('viridis')

    for layer_name in layer_names:
        if layer_name in activations:
            act = activations[layer_name].squeeze(0) # [Channels, H, W]
            num_channels = act.shape[0]
            size = act.shape[1]
            
            st.caption(f"**Layer {layer_name}** (Native: {size}x{size} -> Display: 128x128)")
            
            # Show first 8 filters
            cols = st.columns(8)
            for i in range(8):
                if i < num_channels:
                    with cols[i]:
                        # Normalize 0-1
                        channel_image = act[i]
                        channel_image -= channel_image.min()
                        if channel_image.max() > 0:
                            channel_image /= channel_image.max()
                        
                        img_np = channel_image.cpu().numpy()
                        
                        # Apply Colormap: (H, W) -> (H, W, 4) RGBA Floats (0.0-1.0)
                        colored_img = cmap(img_np)
                        
                        # Convert to Uint8 (0-255) for PIL
                        colored_img_uint8 = (colored_img * 255).astype(np.uint8)
                        
                        # Create PIL Image
                        pil_img = Image.fromarray(colored_img_uint8)
                        
                        # RESIZE to 128x128
                        # We use Nearest Neighbor to keep the 'features' sharp and pixelated
                        # so you can see exactly what the network is looking at.
                        pil_img = pil_img.resize((128, 128), resample=Image.NEAREST)
                        
                        # Display
                        st.image(pil_img, use_container_width=True, clamp=True)

def get_model_summary(model, input_size=(1, 3, 128, 128)):
    try:
        model_stats = summary(model, input_size=input_size, verbose=0)
        return str(model_stats)
    except Exception as e:
        return f"Error generating summary: {e}"

def get_system_metrics():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    gpu_info = "N/A"
    gpu_mem = "N/A"
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_mem = f"{allocated:.2f}GB / {reserved:.2f}GB"
    return cpu, ram, gpu_info, gpu_mem

def stop_execution():
    config.ALIVE = False    

def imshow(img, title=None):
    img = img * 0.5 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# --- FILE & MODEL MANAGEMENT ---

def get_next_model_filename():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    pattern = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME_BASE}_*.pth")
    existing_files = glob.glob(pattern)
    max_num = 0
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        match = re.search(r'_(\d+)\.pth$', filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    new_filename = f"{config.MODEL_NAME_BASE}_{next_num}.pth"
    return os.path.join(config.MODEL_SAVE_DIR, new_filename)

def save_checkpoint(net, optimizer, epochs, lr, batch_size, metrics, custom_name=None):
    if custom_name and custom_name.strip() != "":
        filename = custom_name.strip()
        if not filename.endswith('.pth'):
            filename += '.pth'
        save_path = os.path.join(config.MODEL_SAVE_DIR, filename)
    else:
        save_path = get_next_model_filename()
        
    trackers = st.session_state.get('trackers', {})
    
    plotting_history = {
        'stepTracker': list(trackers.get('stepTracker', [])),
        'batchTracker': list(trackers.get('batchTracker', [])),
        'val_step': list(trackers.get('val_step', [])),
        'val_accuracy': list(trackers.get('val_accuracy', [])),
        'val_batch_idx': int(trackers.get('val_batch_idx', 0)),
        'cm_true': list(trackers.get('cm_true', [])),
        'cm_pred': list(trackers.get('cm_pred', []))
    }

    custom_test_path = st.session_state.get('custom_test_path', None)
    custom_train_path = st.session_state.get('custom_train_path', None)
    custom_val_path = st.session_state.get('custom_val_path', None)

    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'custom_test_path': custom_test_path,
            'custom_train_path': custom_train_path,
            'custom_val_path': custom_val_path
        },
        'metrics': metrics,
        'plotting_history': plotting_history
    }
    
    torch.save(checkpoint, save_path)
    
    st.session_state['last_loaded_path'] = save_path
    st.session_state['model_is_ready'] = True
    st.session_state['active_model_meta'] = checkpoint
    
    st.success(f"Model saved to {save_path}")

def rename_model(old_filename, new_name):
    if not new_name:
        return None, "New name cannot be empty."
    if not new_name.endswith('.pth'):
        new_name += '.pth'
    old_path = os.path.join(config.MODEL_SAVE_DIR, old_filename)
    new_path = os.path.join(config.MODEL_SAVE_DIR, new_name)
    if not os.path.exists(old_path):
        return None, "Original file not found."
    if os.path.exists(new_path):
        return None, "A file with that name already exists."
    try:
        os.rename(old_path, new_path)
        return new_path, None 
    except Exception as e:
        return None, str(e)

def load_specific_model(file_path):
    if os.path.exists(file_path):
        try:
            loaded_data = torch.load(file_path, map_location=config.DEVICE, weights_only=False)
            state_dict = None
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                state_dict = loaded_data['model_state_dict']
                st.session_state['active_model_meta'] = loaded_data
                history = loaded_data.get('plotting_history', {})
                trackers = st.session_state['trackers']
                
                trackers['stepTracker'] = history.get('stepTracker', [])
                trackers['batchTracker'] = history.get('batchTracker', [])
                trackers['val_step'] = history.get('val_step', [])
                trackers['val_accuracy'] = history.get('val_accuracy', [])
                trackers['val_batch_idx'] = history.get('val_batch_idx', 0)
                trackers['cm_true'] = history.get('cm_true', [])
                trackers['cm_pred'] = history.get('cm_pred', [])

                hyperparams = loaded_data.get('hyperparameters', {})
                st.session_state['custom_test_path'] = hyperparams.get('custom_test_path', None)
                st.session_state['custom_train_path'] = hyperparams.get('custom_train_path', None)
                st.session_state['custom_val_path'] = hyperparams.get('custom_val_path', None)

            else:
                state_dict = loaded_data
                st.session_state['active_model_meta'] = {}
                st.session_state['custom_test_path'] = None
                st.session_state['custom_train_path'] = None
                st.session_state['custom_val_path'] = None

            saved_num_classes = state_dict['fc3.weight'].shape[0]
            current_num_classes = st.session_state['net'].fc3.out_features
            
            if saved_num_classes != current_num_classes:
                st.warning(f"⚠️ Resizing Network: Current ({current_num_classes}) -> Saved ({saved_num_classes}) classes.")
                gc.collect()
                new_net = Net(num_classes=saved_num_classes).to(config.DEVICE)
                st.session_state['net'] = new_net
                
                saved_lr = config.LEARNING_RATE
                if 'hyperparameters' in st.session_state['active_model_meta']:
                    saved_lr = st.session_state['active_model_meta']['hyperparameters'].get('learning_rate', config.LEARNING_RATE)
                st.session_state['optimizer'] = optim.Adam(new_net.parameters(), lr=saved_lr)
            
            st.session_state['net'].load_state_dict(state_dict)
            st.session_state['last_loaded_path'] = file_path
            st.session_state['model_is_ready'] = True 
            
            st.toast(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            st.error(f"Error loading model file: {e}")
    else:
        st.error(f"File not found: {file_path}")

def display_active_model_info():
    meta = st.session_state.get('active_model_meta', {})
    if meta:
        hp = meta.get('hyperparameters', {})
        metrics = meta.get('metrics', {})
        st.sidebar.markdown("### 📜 Loaded Model Info")
        st.sidebar.info(f"**Epochs:** {hp.get('epochs', '-')}")
        st.sidebar.info(f"**Batch:** {hp.get('batch_size', '-')}")
        st.sidebar.info(f"**LR:** {hp.get('learning_rate', '-')}")
        
        c_train = hp.get('custom_train_path', None)
        c_val = hp.get('custom_val_path', None)
        c_test = hp.get('custom_test_path', None)
        
        if c_train: st.sidebar.caption(f"**Train:** ...{os.path.basename(c_train)}")
        if c_val: st.sidebar.caption(f"**Valid:** ...{os.path.basename(c_val)}")
        if c_test: st.sidebar.caption(f"**Test:** ...{os.path.basename(c_test)}")

        if 'final_loss' in metrics:
            st.sidebar.metric("Saved Loss", f"{metrics['final_loss']:.4f}")
        if 'val_accuracy' in metrics:
            st.sidebar.metric("Saved Accuracy", f"{metrics['val_accuracy']:.2f}%")
        st.sidebar.markdown("---")