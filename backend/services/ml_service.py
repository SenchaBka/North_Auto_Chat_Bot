import tensorflow as tf
from tensorflow.keras.applications import InceptionV3 # Main GoogLeNet model
from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import json


COMPCARS_ROOT = r'C:\Users\301335235\Desktop\project\dataset\data\data'
IMAGE_SIZE = (299, 299)  # Standard input size for InceptionV3
BATCH_SIZE = 32          # Reduced for local training (increase for cloud GPU)
AUTOTUNE = tf.data.AUTOTUNE

# Paths to the official split lists (from misc/train_test_split/classification/)
TRAIN_LIST = os.path.join(COMPCARS_ROOT, 'train_test_split/classification/train.txt')
TEST_LIST = os.path.join(COMPCARS_ROOT, 'train_test_split/classification/test.txt')

# Model save directory (create if not exists)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Global label mapping (sparse model_ids -> contiguous 0-based labels)
# Will be populated by build_label_mapping()
LABEL_MAPPING = {}
NUM_CLASSES = 0  # Will be set dynamically based on actual dataset

def build_label_mapping():
    """
    Build a mapping from sparse model_ids to contiguous labels.
    CompCars model_ids are sparse (1 to 1993 with only 431 unique values).
    We need contiguous labels 0-430 for the neural network.
    """
    global LABEL_MAPPING, NUM_CLASSES
    
    print("Building label mapping from training data...")
    
    # Collect all unique model_ids from both train and test
    all_model_ids = set()
    
    for list_path in [TRAIN_LIST, TEST_LIST]:
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('/')
                    if len(parts) >= 2:
                        model_id = int(parts[1])
                        all_model_ids.add(model_id)
    
    # Sort and create mapping: sparse model_id -> contiguous label
    sorted_model_ids = sorted(all_model_ids)
    LABEL_MAPPING = {model_id: idx for idx, model_id in enumerate(sorted_model_ids)}
    NUM_CLASSES = len(LABEL_MAPPING)
    
    print(f"  Found {NUM_CLASSES} unique car models")
    print(f"  Model ID range: {min(sorted_model_ids)} to {max(sorted_model_ids)}")
    
    # Save mapping for inference later
    mapping_path = os.path.join(MODELS_DIR, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in LABEL_MAPPING.items()}, f)
    print(f"  Label mapping saved to: {mapping_path}")
    
    return LABEL_MAPPING

# Enable Mixed Precision (FP16) for modern NVIDIA GPUs (Vast.ai performance boost)
# Note: Disable if no GPU or older GPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision (FP16) enabled for GPU acceleration")
    else:
        print("No GPU detected - using default float32 precision")
except Exception as e:
    print(f"Could not configure mixed precision: {e}")

# Preprocess the data

def parse_split_file(list_path):
    """
    Parse CompCars split file and return image paths with mapped labels.
    
    Each line format: 'make_id/model_id/released_year/image_name.jpg'
    Labels are mapped from sparse model_ids to contiguous 0-based indices.
    """
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"CompCars list file not found at: {list_path}")

    with open(list_path, 'r') as f:
        lines = f.readlines()
    
    image_paths = []
    labels = []
    skipped = 0
    
    for line in lines:
        relative_path = line.strip()
        if not relative_path:
            continue
            
        parts = relative_path.split('/')
        
        # We use model_id (second component) as the classification label
        if len(parts) >= 2:
            model_id = int(parts[1])
            
            # Map sparse model_id to contiguous label using global mapping
            if model_id in LABEL_MAPPING:
                image_paths.append(os.path.join(COMPCARS_ROOT, 'image', relative_path))
                labels.append(LABEL_MAPPING[model_id])
            else:
                skipped += 1
    
    if skipped > 0:
        print(f"  Warning: Skipped {skipped} samples with unknown model_ids")
    
    return np.array(image_paths), np.array(labels, dtype=np.int32)

def load_and_preprocess_image(image_path, label):
    
    # 1. Load Image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # 2. Resize and Preprocess (Bilinear interpolation is default for tf.image.resize)
    img = tf.image.resize(img, IMAGE_SIZE)
    
    # 3. InceptionV3 specific preprocessing: Scales pixels to the range [-1, 1]
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    # 4. One-Hot Encode Label
    label = tf.one_hot(label, NUM_CLASSES)
    
    return img, label

def create_compcars_dataset(list_path):
    # Use tf.Dataset to load the data
    paths, labels = parse_split_file(list_path)
    
    # 1. Create a Dataset from the parsed arrays
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    # 2. Map the preprocessing function using parallelism
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    
    # 3. Configure Dataset for Performance (Shuffle, Batch, Cache, Prefetch)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    # .cache() keeps images in memory after first epoch if possible, speeds up subsequent epochs
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE) 

    return dataset

# Model Training (will be performed on vast.ai)

def build_fine_tuning_model(input_shape, num_classes):
    
    # Load InceptionV3 base model (pre-trained on ImageNet)
    base_model = InceptionV3(
        include_top=False,  # Exclude the original classification head
        weights='imagenet', # Use pre-trained weights
        input_shape=input_shape
    )
    
    # 1. Freeze the entire base model for Phase 1
    base_model.trainable = False 
    
    # 2. Define the New Classification Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x) # Reduces (8, 8, 2048) to (2048)
    
    # Add a Dropout Layer for regularization (prevents overfitting to small training data on top)
    x = layers.Dropout(0.2)(x) 
    
    # The final classification layer MUST output float32 (stability in mixed_precision)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model, base_model

def train_model():
    
    # Build label mapping first (sets NUM_CLASSES)
    build_label_mapping()
    
    print("\nInitializing Datasets from official CompCars splits...")
    train_ds = create_compcars_dataset(TRAIN_LIST)
    test_ds = create_compcars_dataset(TEST_LIST)

    # --- Model Setup ---
    print(f"\nBuilding model with {NUM_CLASSES} output classes...")
    model, base_model = build_fine_tuning_model(IMAGE_SIZE + (3,), NUM_CLASSES)
    
    # --- Callbacks ---
    # Save the best model only (will be overwritten in Phase 2)
    checkpoint_filepath = os.path.join(MODELS_DIR, 'temp_best_model.keras')
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    
    # Train new classification head
    
    print("\n--- PHASE 1: Training Classification Head (Base Model Frozen) ---")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Higher LR for new layers
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Initial quick training to establish the new dense layer weights
    model.fit(
        train_ds,
        epochs=5, 
        validation_data=test_ds,
        callbacks=[model_checkpoint_callback]
    )

    # Fine-tune the top layers of InceptionV3
    
    # 1. Unfreeze the entire base model
    base_model.trainable = True

    # 2. Selectively freeze layers: Only keep the top layers unfrozen
    # Unfreeze only the last 50 layers for fine-tuning, keeping initial layers frozen for stability
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    # 3. Recompile with a VASTLY lower learning rate (CRITICAL for fine-tuning)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Very low LR for small adjustments
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add an Early Stopping callback for efficiency on Vast.ai
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    # Train the model further, allowing small weight updates in the top Inception blocks
    history = model.fit(
        train_ds,
        epochs=30, # Max epochs, but EarlyStopping will likely halt it sooner
        validation_data=test_ds,
        callbacks=[early_stopping_callback]
    )
    
    # --- Model Persistence ---
    final_model_path = os.path.join(MODELS_DIR, 'inceptionv3_compcars.keras')
    model.save(final_model_path)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Label mapping saved to: {os.path.join(MODELS_DIR, 'label_mapping.json')}")
    print(f"{'='*60}")
    
    # Return history object for logging/metrics display in the frontend
    return model, history


def validate_setup():
    """Validate the setup and test model building without requiring full dataset."""
    print("=" * 60)
    print("VALIDATING SETUP AND TESTING MODEL")
    print("=" * 60)
    
    # 1. Check TensorFlow version and GPU
    print("\n[1/6] Checking TensorFlow and GPU...")
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
    else:
        print("[WARNING] No GPU detected. Training will be slow on CPU.")
    
    # 2. Check dataset root directory
    print("\n[2/6] Checking dataset directory...")
    if os.path.exists(COMPCARS_ROOT):
        print(f"[OK] Dataset root exists: {COMPCARS_ROOT}")
    else:
        print(f"[ERROR] Dataset root NOT found: {COMPCARS_ROOT}")
        return False
    
    # 3. Check train/test split files
    print("\n[3/6] Checking train/test split files...")
    train_exists = os.path.exists(TRAIN_LIST)
    test_exists = os.path.exists(TEST_LIST)
    
    if train_exists:
        print(f"[OK] Train list found: {TRAIN_LIST}")
        try:
            with open(TRAIN_LIST, 'r') as f:
                train_count = sum(1 for _ in f)
            print(f"  - Contains {train_count} training samples")
        except Exception as e:
            print(f"  - Error reading file: {e}")
    else:
        print(f"[ERROR] Train list NOT found: {TRAIN_LIST}")
    
    if test_exists:
        print(f"[OK] Test list found: {TEST_LIST}")
        try:
            with open(TEST_LIST, 'r') as f:
                test_count = sum(1 for _ in f)
            print(f"  - Contains {test_count} test samples")
        except Exception as e:
            print(f"  - Error reading file: {e}")
    else:
        print(f"[ERROR] Test list NOT found: {TEST_LIST}")
    
    if not (train_exists and test_exists):
        print("\n[WARNING] Train/test split files not found.")
        print("  The dataset structure may be incomplete.")
        print("  Model building will be tested, but training cannot proceed.")
        return False
    
    # 4. Check image directory and sample images
    print("\n[4/6] Checking image directory...")
    image_dir = os.path.join(COMPCARS_ROOT, 'image')
    if os.path.exists(image_dir):
        print(f"[OK] Image directory exists: {image_dir}")
        # Verify a few sample images exist
        with open(TRAIN_LIST, 'r') as f:
            sample_lines = [l.strip() for l in f.readlines()[:5]]
        missing = 0
        for line in sample_lines:
            img_path = os.path.join(COMPCARS_ROOT, 'image', line)
            if not os.path.exists(img_path):
                missing += 1
        if missing == 0:
            print(f"  - Sample images verified: OK")
        else:
            print(f"  - WARNING: {missing}/5 sample images missing!")
    else:
        print(f"[ERROR] Image directory NOT found: {image_dir}")
        return False
    
    # 5. Build label mapping
    print("\n[5/6] Building label mapping...")
    try:
        build_label_mapping()
        print(f"[OK] Label mapping built: {NUM_CLASSES} classes")
    except Exception as e:
        print(f"[ERROR] Failed to build label mapping: {e}")
        return False
    
    # 6. Test model building
    print("\n[6/6] Testing model building...")
    try:
        print(f"  Building InceptionV3 model with {NUM_CLASSES} classes...")
        model, base_model = build_fine_tuning_model(IMAGE_SIZE + (3,), NUM_CLASSES)
        
        print("  [OK] Model built successfully!")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Output shape: {model.output_shape}")
        print(f"  - Total parameters: {model.count_params():,}")
        print(f"  - Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        print(f"  - Non-trainable parameters: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
        
        # Test compilation
        print("  Testing model compilation...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("  [OK] Model compiled successfully!")
        
        # Test forward pass with dummy data
        print("  Testing forward pass with dummy data...")
        dummy_input = tf.random.normal((1, *IMAGE_SIZE, 3))
        dummy_output = model(dummy_input, training=False)
        print(f"  [OK] Forward pass successful! Output shape: {dummy_output.shape}")
        print(f"  - Output sum (should be ~1.0 for softmax): {tf.reduce_sum(dummy_output).numpy():.4f}")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED! Setup is ready for training.")
        print(f"  - Dataset: {train_count + test_count} images, {NUM_CLASSES} classes")
        print(f"  - Models will be saved to: {MODELS_DIR}")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"  [ERROR] Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    args = sys.argv[1:]
    test_mode = '--test' in args
    force_train = '--yes' in args or '-y' in args
    
    if test_mode:
        # Run validation tests only
        success = validate_setup()
        sys.exit(0 if success else 1)
    
    # Otherwise, run full training
    try:
        # First run validation
        print("Running setup validation before training...\n")
        if not validate_setup():
            print("\n[WARNING] Setup validation failed. Please fix the issues above.")
            print("You can run with --test flag to validate setup only: python ml_service.py --test")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("STARTING FULL TRAINING")
        print("=" * 60 + "\n")
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("WARNING: No GPU detected. Training will be slow on CPU.")
            if not force_train:
                user_input = input("Continue anyway? (y/n): ").strip().lower()
                if user_input != 'y':
                    print("Training cancelled.")
                    sys.exit(0)
            else:
                print("Proceeding with CPU training (--yes flag provided)...")
        else:
            print(f"Using GPU: {gpus[0].name}")

        # If using MirroredStrategy for multi-GPU, uncomment the following:
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        #     trained_model, training_history = train_model()

        trained_model, training_history = train_model()
        
    except FileNotFoundError as e:
        print(f"\n[FATAL ERROR] {e}")
        print("Please ensure the CompCars dataset is extracted and paths are correct.")
        print("\nRun with --test flag to validate setup: python ml_service.py --test")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)