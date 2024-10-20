import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split  # Add this import

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Number of classes for output
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage
if __name__ == "__main__":
    from data_preprocessing import load_data

    # Load data
    data_dir = '/path/to/your/dataset'  # Replace with your dataset path
    X, y, class_names = load_data(data_dir)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1:], len(class_names))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])

    # Save the model
    model.save('classification_model.h5')
    print("Model training complete and saved as 'classification_model.h5'.")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
