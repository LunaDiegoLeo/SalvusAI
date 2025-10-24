"""
SALVUS AI - Sistema de Detección Médica
Interfaz Gráfica Principal
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from datetime import datetime

class SalvusAI:
    def __init__(self, root):
        self.root = root
        self.root.title("Salvus AI - Sistema de Detección Médica")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0a0e27")
        
        # Variables
        self.modelos = {}
        self.clases = {}
        self.imagen_actual = None
        self.tipo_actual = None
        
        # Cargar modelos
        self.cargar_modelos()
        
        # Crear interfaz
        self.crear_interfaz()
        
    def cargar_modelos(self):
        """Carga los modelos entrenados"""
        tipos = ['pulmones', 'piel', 'cerebro']
        for tipo in tipos:
            try:
                modelo_path = f'modelo_{tipo}_final.h5'
                clases_path = f'clases_{tipo}.json'
                
                if os.path.exists(modelo_path) and os.path.exists(clases_path):
                    self.modelos[tipo] = keras.models.load_model(modelo_path)
                    with open(clases_path, 'r', encoding='utf-8') as f:
                        self.clases[tipo] = json.load(f)
                    print(f"✓ Modelo {tipo} cargado correctamente")
                else:
                    print(f"⚠ Modelo {tipo} no encontrado")
            except Exception as e:
                print(f"✗ Error cargando modelo {tipo}: {e}")
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#1a1f3a", height=120)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Logo y título
        title_frame = tk.Frame(header_frame, bg="#1a1f3a")
        title_frame.pack(expand=True)
        
        # Título principal
        title_label = tk.Label(
            title_frame,
            text="⚕ SALVUS AI",
            font=("Arial", 42, "bold"),
            fg="#00d4ff",
            bg="#1a1f3a"
        )
        title_label.pack(pady=(10, 0))
        
        # Subtítulo
        subtitle_label = tk.Label(
            title_frame,
            text="Sistema Inteligente de Detección Médica | Deep Learning CNN",
            font=("Arial", 14),
            fg="#7a8bff",
            bg="#1a1f3a"
        )
        subtitle_label.pack()
        
        # Contenedor principal
        main_container = tk.Frame(self.root, bg="#0a0e27")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Panel izquierdo - Botones de selección
        left_panel = tk.Frame(main_container, bg="#141937", width=350)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Título del panel
        tk.Label(
            left_panel,
            text="SELECCIONA EL ANÁLISIS",
            font=("Arial", 16, "bold"),
            fg="#ffffff",
            bg="#141937"
        ).pack(pady=(30, 20))
        
        # Botones de selección
        self.crear_boton_categoria(
            left_panel,
            "🫁 ANÁLISIS DE PULMONES",
            "Radiografías de tórax",
            "#ff6b6b",
            lambda: self.seleccionar_imagen('pulmones')
        )
        
        self.crear_boton_categoria(
            left_panel,
            "🩺 ANÁLISIS DE PIEL",
            "Fotografías de lesiones",
            "#4ecdc4",
            lambda: self.seleccionar_imagen('piel')
        )
        
        self.crear_boton_categoria(
            left_panel,
            "🧠 ANÁLISIS DE CEREBRO",
            "Tomografías cerebrales",
            "#a78bfa",
            lambda: self.seleccionar_imagen('cerebro')
        )
        
        # Información del sistema
        info_frame = tk.Frame(left_panel, bg="#1a1f3a", relief="solid", bd=1)
        info_frame.pack(side="bottom", fill="x", padx=15, pady=15)
        
        tk.Label(
            info_frame,
            text="MODELOS CARGADOS",
            font=("Arial", 10, "bold"),
            fg="#7a8bff",
            bg="#1a1f3a"
        ).pack(pady=(10, 5))
        
        for tipo in ['pulmones', 'piel', 'cerebro']:
            estado = "✓ Activo" if tipo in self.modelos else "✗ No disponible"
            color = "#00ff88" if tipo in self.modelos else "#ff4444"
            tk.Label(
                info_frame,
                text=f"{tipo.capitalize()}: {estado}",
                font=("Arial", 9),
                fg=color,
                bg="#1a1f3a"
            ).pack(anchor="w", padx=15, pady=2)
        
        tk.Label(
            info_frame,
            text="",
            bg="#1a1f3a"
        ).pack(pady=5)
        
        # Panel derecho - Visualización y resultados
        right_panel = tk.Frame(main_container, bg="#141937")
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Área de visualización de imagen
        self.image_frame = tk.Frame(right_panel, bg="#1a1f3a", relief="solid", bd=2)
        self.image_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Label para la imagen
        self.image_label = tk.Label(
            self.image_frame,
            text="Selecciona un tipo de análisis y carga una imagen",
            font=("Arial", 16),
            fg="#7a8bff",
            bg="#1a1f3a"
        )
        self.image_label.pack(expand=True)
        
        # Panel de resultados
        self.results_frame = tk.Frame(right_panel, bg="#1a1f3a", relief="solid", bd=2)
        self.results_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        tk.Label(
            self.results_frame,
            text="RESULTADOS DEL ANÁLISIS",
            font=("Arial", 14, "bold"),
            fg="#00d4ff",
            bg="#1a1f3a"
        ).pack(pady=15)
        
        self.results_text = tk.Text(
            self.results_frame,
            height=8,
            font=("Courier New", 11),
            bg="#0f1424",
            fg="#ffffff",
            relief="flat",
            padx=20,
            pady=15
        )
        self.results_text.pack(fill="x", padx=15, pady=(0, 15))
        self.results_text.insert("1.0", "No hay resultados todavía. Carga una imagen para comenzar el análisis.")
        self.results_text.config(state="disabled")
        
    def crear_boton_categoria(self, parent, texto, subtexto, color, comando):
        """Crea un botón estilizado para cada categoría"""
        btn_frame = tk.Frame(parent, bg=color, relief="flat", cursor="hand2")
        btn_frame.pack(fill="x", padx=15, pady=10)
        
        btn = tk.Frame(btn_frame, bg=color)
        btn.pack(fill="both", expand=True, padx=3, pady=3)
        
        tk.Label(
            btn,
            text=texto,
            font=("Arial", 13, "bold"),
            fg="#ffffff",
            bg=color
        ).pack(pady=(15, 5))
        
        tk.Label(
            btn,
            text=subtexto,
            font=("Arial", 9),
            fg="#ffffff",
            bg=color
        ).pack(pady=(0, 15))
        
        # Bind events
        btn_frame.bind("<Button-1>", lambda e: comando())
        btn.bind("<Button-1>", lambda e: comando())
        for child in btn.winfo_children():
            child.bind("<Button-1>", lambda e: comando())
        
        # Hover effects
        def on_enter(e):
            btn_frame.config(bg=self.adjust_color(color, 1.2))
            btn.config(bg=self.adjust_color(color, 1.2))
            for child in btn.winfo_children():
                child.config(bg=self.adjust_color(color, 1.2))
        
        def on_leave(e):
            btn_frame.config(bg=color)
            btn.config(bg=color)
            for child in btn.winfo_children():
                child.config(bg=color)
        
        btn_frame.bind("<Enter>", on_enter)
        btn_frame.bind("<Leave>", on_leave)
    
    def adjust_color(self, color, factor):
        """Ajusta el brillo de un color"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c * factor)) for c in rgb)
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    
    def seleccionar_imagen(self, tipo):
        """Abre el diálogo para seleccionar una imagen"""
        if tipo not in self.modelos:
            messagebox.showerror(
                "Error",
                f"El modelo para {tipo} no está disponible.\nPor favor, entrena el modelo primero."
            )
            return
        
        file_path = filedialog.askopenfilename(
            title=f"Seleccionar imagen para análisis de {tipo}",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            self.tipo_actual = tipo
            self.cargar_y_analizar_imagen(file_path)
    
    def cargar_y_analizar_imagen(self, file_path):
        """Carga la imagen y realiza el análisis"""
        try:
            # Cargar imagen original
            img = Image.open(file_path)
            self.imagen_actual = img.copy()
            
            # Mostrar imagen
            self.mostrar_imagen(img)
            
            # Realizar predicción
            self.predecir(img)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen:\n{e}")
    
    def mostrar_imagen(self, img):
        """Muestra la imagen en la interfaz"""
        # Redimensionar manteniendo aspecto
        display_size = (700, 500)
        img.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Crear imagen para mostrar
        photo = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def predecir(self, img):
        """Realiza la predicción con el modelo"""
        try:
            # Preparar imagen
            img_array = img.resize((224, 224))
            img_array = np.array(img_array)
            
            # Si la imagen es en escala de grises, convertir a RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Realizar predicción
            modelo = self.modelos[self.tipo_actual]
            prediccion = modelo.predict(img_array, verbose=0)
            
            # Obtener clase predicha y confianza
            clase_idx = np.argmax(prediccion[0])
            confianza = prediccion[0][clase_idx] * 100
            
            # Invertir diccionario de clases
            clases_inv = {v: k for k, v in self.clases[self.tipo_actual].items()}
            clase_nombre = clases_inv[clase_idx]
            
            # Obtener top 3 predicciones
            top_3_idx = np.argsort(prediccion[0])[-3:][::-1]
            top_3 = [(clases_inv[i], prediccion[0][i] * 100) for i in top_3_idx]
            
            # Mostrar resultados
            self.mostrar_resultados(clase_nombre, confianza, top_3)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar la predicción:\n{e}")
    
    def mostrar_resultados(self, clase, confianza, top_3):
        """Muestra los resultados del análisis"""
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        
        # Fecha y hora
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        resultado = f"""
╔══════════════════════════════════════════════════════════════════╗
║  ANÁLISIS DE {self.tipo_actual.upper()}
║  Fecha: {now}
╚══════════════════════════════════════════════════════════════════╝

DIAGNÓSTICO PRINCIPAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🎯 {clase}
  📊 Confianza: {confianza:.2f}%
  {'🟢 ALTA CONFIANZA' if confianza > 80 else '🟡 CONFIANZA MEDIA' if confianza > 60 else '🔴 BAJA CONFIANZA'}


PROBABILIDADES DETALLADAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        for i, (nombre, prob) in enumerate(top_3, 1):
            barra = "█" * int(prob / 2) + "░" * (50 - int(prob / 2))
            resultado += f"  {i}. {nombre}\n"
            resultado += f"     [{barra}] {prob:.2f}%\n\n"
        
        resultado += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  AVISO IMPORTANTE:
Este sistema es una herramienta de apoyo diagnóstico. Los resultados
deben ser validados por un profesional médico calificado.
"""
        
        self.results_text.insert("1.0", resultado)
        self.results_text.config(state="disabled")


# ==============================================================================
# EJECUTAR APLICACIÓN
# ==============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = SalvusAI(root)
    root.mainloop()
