"""
SALVUS AI - Sistema de Detección Médica con CNN
Módulo de Entrenamiento de Modelos
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os

class SalvusModelTrainer:
    def __init__(self, model_type, img_size=(224, 224)):
        """
        model_type: 'pulmones', 'piel', o 'cerebro'
        img_size: tamaño de las imágenes
        """
        self.model_type = model_type
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def create_cnn_model(self, num_classes):
        """Crea una CNN optimizada para detección médica"""
        model = models.Sequential([
            # Bloque 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Capas densas
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def prepare_data(self, data_dir, validation_split=0.2, batch_size=32):
        """Prepara los datos con data augmentation"""
        
        # Data augmentation para entrenamiento
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Solo rescale para validación
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Generadores de datos
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, data_dir, epochs=50, batch_size=32):
        """Entrena el modelo"""
        
        print(f"\n{'='*60}")
        print(f"ENTRENANDO MODELO: {self.model_type.upper()}")
        print(f"{'='*60}\n")
        
        # Preparar datos
        train_gen, val_gen = self.prepare_data(data_dir, batch_size=batch_size)
        num_classes = len(train_gen.class_indices)
        
        print(f"Clases detectadas: {list(train_gen.class_indices.keys())}")
        print(f"Número de clases: {num_classes}")
        print(f"Imágenes de entrenamiento: {train_gen.samples}")
        print(f"Imágenes de validación: {val_gen.samples}\n")
        
        # Crear modelo
        self.model = self.create_cnn_model(num_classes)
        
        # Compilar
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'modelo_{self.model_type}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar modelo final y clases
        self.model.save(f'modelo_{self.model_type}_final.h5')
        
        # Guardar nombres de clases
        import json
        with open(f'clases_{self.model_type}.json', 'w', encoding='utf-8') as f:
            json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Modelo guardado como: modelo_{self.model_type}_final.h5")
        print(f"✓ Clases guardadas como: clases_{self.model_type}.json")
        
        # Mostrar resultados
        self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """Visualiza el historial de entrenamiento"""
        if self.history is None:
            print("No hay historial de entrenamiento disponible")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Historial de Entrenamiento - {self.model_type.upper()}', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Entrenamiento')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validación')
        axes[0, 0].set_title('Precisión del Modelo')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Entrenamiento')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validación')
        axes[0, 1].set_title('Pérdida del Modelo')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Entrenamiento')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validación')
        axes[1, 0].set_title('Precisión')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Entrenamiento')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validación')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'entrenamiento_{self.model_type}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráficas guardadas como: entrenamiento_{self.model_type}.png")
        plt.show()


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    """
    ESTRUCTURA DE CARPETAS ESPERADA:
    
    datos_pulmones/
        ├── pulmones_sanos/
        │   ├── img1.jpg
        │   ├── img2.jpg
        ├── neumonia/
        ├── tuberculosis/
        └── covid/
    
    datos_piel/
        ├── queratosis_actinica/
        ├── carcinoma_basocelular/
        ├── dermatofibroma/
        ├── melanoma/
        ├── lunar/
        ├── queratosis_benigna_pigmentada/
        ├── queratosis_seborreica/
        ├── carcinoma_escamocelular/
        └── lesion_vascular/
    
    datos_cerebro/
        ├── glioma/
        ├── meningioma/
        ├── adenoma/
        └── no_tumor/
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                    SALVUS AI                             ║
    ║          Sistema de Detección Médica con CNN             ║
    ║                                                          ║
    ║  Entrenamiento de Modelos de Deep Learning              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # ENTRENAR MODELO DE PULMONES
    print("\n1️⃣  MODELO DE PULMONES")
    trainer_pulmones = SalvusModelTrainer('pulmones', img_size=(224, 224))
    trainer_pulmones.train('datos_pulmones', epochs=50, batch_size=32)
    
    # ENTRENAR MODELO DE PIEL
    print("\n2️⃣  MODELO DE PIEL")
    trainer_piel = SalvusModelTrainer('piel', img_size=(224, 224))
    trainer_piel.train('datos_piel', epochs=50, batch_size=32)
    
    # ENTRENAR MODELO DE CEREBRO
    print("\n3️⃣  MODELO DE CEREBRO")
    trainer_cerebro = SalvusModelTrainer('cerebro', img_size=(224, 224))
    trainer_cerebro.train('datos_cerebro', epochs=50, batch_size=32)
    
    print("\n" + "="*60)
    print("✓ TODOS LOS MODELOS HAN SIDO ENTRENADOS EXITOSAMENTE")
    print("="*60)
