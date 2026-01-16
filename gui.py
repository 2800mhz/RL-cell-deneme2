# gui.py - Gelişmiş PyQt5 Arayüzü

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from environment import CellEnv
from stable_baselines3 import PPO

class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib Canvas"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

class CellSimulationGUI(QtWidgets.QMainWindow):
    """
    Gelişmiş hücre simülasyonu arayüzü
    """
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("🧬 Gelişmiş Hücre Simülasyonu - RL")
        self.setGeometry(100, 100, 1400, 800)
        
        # Ortam ve model
        self.env = CellEnv(max_steps=1000)
        self.model = None
        self.obs = self.env.reset()
        
        # Veri geçmişi
        self.history = {
            'step': [],
            'size': [],
            'atp': [],
            'glucose': [],
            'amino_acids': [],
            'fatty_acids': [],
            'oxygen': [],
            'damage': [],
            'stress': [],
            'total_waste': [],
            'growth_points': []
        }
        
        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.auto_step)
        self.is_running = False
        self.step_interval = 100  # ms
        
        # UI oluştur
        self.init_ui()
        
        # İlk güncelleme
        self.update_display()
    
    def init_ui(self):
        """UI bileşenlerini oluştur"""
        # Ana widget
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        
        # Ana layout
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        
        # SOL PANEL: Kontroller ve Bilgiler
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # SAĞ PANEL: Grafikler
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=3)
    
    def create_left_panel(self):
        """Sol kontrol paneli"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Başlık
        title = QtWidgets.QLabel("🧬 HÜCRE DURUMU")
        title.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # Durum bilgileri
        self.status_group = QtWidgets.QGroupBox("Genel Durum")
        status_layout = QtWidgets.QVBoxLayout()
        
        self.lbl_alive = QtWidgets.QLabel("Durum: CANLI")
        self.lbl_alive.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.lbl_size = QtWidgets.QLabel("Boyut: 1.00")
        self.lbl_age = QtWidgets.QLabel("Yaş: 0")
        self.lbl_metabolic = QtWidgets.QLabel("Metabolik Hız: 1.00x")
        
        status_layout.addWidget(self.lbl_alive)
        status_layout.addWidget(self.lbl_size)
        status_layout.addWidget(self.lbl_age)
        status_layout.addWidget(self.lbl_metabolic)
        self.status_group.setLayout(status_layout)
        layout.addWidget(self.status_group)
        
        # Enerji ve kaynaklar
        self.energy_group = QtWidgets.QGroupBox("Enerji & Kaynaklar")
        energy_layout = QtWidgets.QVBoxLayout()
        
        self.lbl_atp = QtWidgets.QLabel("ATP: 0")
        self.lbl_glucose = QtWidgets.QLabel("Glikoz: 0")
        self.lbl_amino = QtWidgets.QLabel("Amino Asit: 0")
        self.lbl_fatty = QtWidgets.QLabel("Yağ Asidi: 0")
        self.lbl_oxygen = QtWidgets.QLabel("Oksijen: 0")
        self.lbl_water = QtWidgets.QLabel("Su: 0")
        
        for lbl in [self.lbl_atp, self.lbl_glucose, self.lbl_amino, 
                    self.lbl_fatty, self.lbl_oxygen, self.lbl_water]:
            energy_layout.addWidget(lbl)
        
        self.energy_group.setLayout(energy_layout)
        layout.addWidget(self.energy_group)
        
        # Sağlık
        self.health_group = QtWidgets.QGroupBox("Sağlık")
        health_layout = QtWidgets.QVBoxLayout()
        
        self.lbl_damage = QtWidgets.QLabel("Hasar: 0%")
        self.lbl_stress = QtWidgets.QLabel("Stres: 0%")
        self.lbl_waste = QtWidgets.QLabel("Atık: 0")
        self.lbl_growth = QtWidgets.QLabel("Büyüme: 0%")
        
        for lbl in [self.lbl_damage, self.lbl_stress, self.lbl_waste, self.lbl_growth]:
            health_layout.addWidget(lbl)
        
        self.health_group.setLayout(health_layout)
        layout.addWidget(self.health_group)
        
        # Kontrol butonları
        self.control_group = QtWidgets.QGroupBox("Kontroller")
        control_layout = QtWidgets.QVBoxLayout()
        
        # Model yükle
        self.btn_load = QtWidgets.QPushButton(" Model Yükle")
        self.btn_load.clicked.connect(self.load_model)
        control_layout.addWidget(self.btn_load)
        
        # Başlat/Durdur
        self.btn_start = QtWidgets.QPushButton(" Başlat")
        self.btn_start.clicked.connect(self.toggle_simulation)
        control_layout.addWidget(self.btn_start)
        
        # Tek adım
        self.btn_step = QtWidgets.QPushButton(" Tek Adım")
        self.btn_step.clicked.connect(self.manual_step)
        control_layout.addWidget(self.btn_step)
        
        # Sıfırla
        self.btn_reset = QtWidgets.QPushButton(" Sıfırla")
        self.btn_reset.clicked.connect(self.reset_simulation)
        control_layout.addWidget(self.btn_reset)
        
        # Hız slider
        speed_label = QtWidgets.QLabel("Simülasyon Hızı:")
        control_layout.addWidget(speed_label)
        
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.speed_slider.setTickInterval(100)
        self.speed_slider.valueChanged.connect(self.change_speed)
        control_layout.addWidget(self.speed_slider)
        
        self.lbl_speed = QtWidgets.QLabel("100 ms/adım")
        self.lbl_speed.setAlignment(QtCore.Qt.AlignCenter)
        control_layout.addWidget(self.lbl_speed)
        
        self.control_group.setLayout(control_layout)
        layout.addWidget(self.control_group)
        
        # Son aksiyon
        self.action_group = QtWidgets.QGroupBox("Son Aksiyon")
        action_layout = QtWidgets.QVBoxLayout()
        self.lbl_action = QtWidgets.QLabel("Henüz aksiyon alınmadı")
        self.lbl_action.setWordWrap(True)
        action_layout.addWidget(self.lbl_action)
        self.action_group.setLayout(action_layout)
        layout.addWidget(self.action_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Sağ grafik paneli"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Grafik canvas
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        layout.addWidget(self.canvas)
        
        return panel
    
    def load_model(self):
        """Model yükle"""
        model_path = "models/cell_model.zip"
        if os.path.exists(model_path):
            try:
                self.model = PPO.load("models/cell_model", env=self.env)
                QtWidgets.QMessageBox.information(
                    self, "Başarılı", 
                    "Model başarıyla yüklendi!\n\nArtık 'Başlat' butonuna basarak\nYZ kontrollü simülasyon izleyebilirsiniz."
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Hata", 
                    f"Model yüklenemedi:\n{str(e)}"
                )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Model Bulunamadı",
                f"Model dosyası bulunamadı:\n{model_path}\n\n"
                "Önce 'python train.py train' komutuyla\nmodeli eğitmelisiniz."
            )
    
    def toggle_simulation(self):
        """Simülasyonu başlat/durdur"""
        if self.is_running:
            self.timer.stop()
            self.is_running = False
            self.btn_start.setText("▶ Başlat")
        else:
            self.timer.start(self.step_interval)
            self.is_running = True
            self.btn_start.setText("⏸ Durdur")
    
    def change_speed(self, value):
        """Simülasyon hızını değiştir"""
        self.step_interval = value
        self.lbl_speed.setText(f"{value} ms/adım")
        if self.is_running:
            self.timer.setInterval(value)
    
    def manual_step(self):
        """Tek adım at"""
        if not self.env.cell.is_alive():
            QtWidgets.QMessageBox.information(
                self, "Hücre Öldü",
                "Hücre artık canlı değil.\nYeni simülasyon için 'Sıfırla' butonuna basın."
            )
            return
        
        self.execute_step()
    
    def auto_step(self):
        """Otomatik adım (timer)"""
        if not self.env.cell.is_alive():
            self.timer.stop()
            self.is_running = False
            self.btn_start.setText("▶ Başlat")
            
            # Özet göster
            max_size = max(self.history['size']) if self.history['size'] else 1.0
            final_age = self.env.cell.age
            
            QtWidgets.QMessageBox.information(
                self, "Simülasyon Bitti",
                f"Hücre öldü!\n\n"
                f"Yaşadığı süre: {final_age} adım\n"
                f"Ulaştığı max boyut: {max_size:.2f}x\n"
                f"{' Bölünmeye ulaştı!' if max_size >= 1.8 else ''}"
            )
            return
        
        self.execute_step()
    
    def execute_step(self):
        """Bir adım çalıştır"""
        # Aksiyon seç (model varsa model, yoksa rastgele)
        if self.model:
            action, _ = self.model.predict(self.obs, deterministic=True)
        else:
            action = self.env.action_space.sample()
        
        # Adım at
        self.obs, reward, done, info = self.env.step(action)
        
        # Aksiyon adını göster
        action_name = self.env.get_action_name(action)
        self.lbl_action.setText(f"{action_name}\n(Ödül: {reward:.2f})")
        
        # Veriyi kaydet
        self.record_data()
        
        # Ekranı güncelle
        self.update_display()
    
    def record_data(self):
        """Veriyi kaydet"""
        state = self.obs
        self.history['step'].append(self.env.current_step)
        self.history['size'].append(state[0])
        self.history['atp'].append(state[1])
        self.history['glucose'].append(state[2])
        self.history['amino_acids'].append(state[3])
        self.history['fatty_acids'].append(state[4])
        self.history['oxygen'].append(state[5])
        self.history['damage'].append(state[12])
        self.history['stress'].append(state[13])
        self.history['total_waste'].append(state[10] + state[11])
        self.history['growth_points'].append(state[14] * 100)
    
    def update_display(self):
        """Ekranı güncelle"""
        cell = self.env.cell
        
        # Durum
        if cell.is_alive():
            self.lbl_alive.setText("✓ DURUM: CANLI")
            self.lbl_alive.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.lbl_alive.setText("✗ DURUM: ÖLÜ")
            self.lbl_alive.setStyleSheet("color: red; font-weight: bold;")
        
        self.lbl_size.setText(f"Boyut: {cell.size:.2f}x")
        self.lbl_age.setText(f"Yaş: {cell.age} adım")
        self.lbl_metabolic.setText(f"Metabolik Hız: {cell.get_metabolic_rate():.2f}x")
        
        # Enerji
        self.lbl_atp.setText(f"⚡ ATP: {cell.atp:.1f}")
        self.lbl_glucose.setText(f"🬠Glikoz: {cell.glucose:.1f}")
        self.lbl_amino.setText(f" Amino Asit: {cell.amino_acids:.1f}")
        self.lbl_fatty.setText(f" Yağ Asidi: {cell.fatty_acids:.1f}")
        self.lbl_oxygen.setText(f" Oksijen: {cell.oxygen:.1f}")
        self.lbl_water.setText(f" Su: {cell.water:.1f}")
        
        # Sağlık
        self.lbl_damage.setText(f" Hasar: {cell.damage:.1f}%")
        self.lbl_stress.setText(f" Stres: {cell.stress_level:.1f}%")
        total_waste = cell.co2 + cell.lactate + cell.urea
        self.lbl_waste.setText(f" Atık: {total_waste:.1f}")
        self.lbl_growth.setText(f" Büyüme: {cell.growth_points:.1f}%")
        
        # Grafik
        self.plot_data()
    
    def plot_data(self):
        """Grafikleri çiz"""
        self.canvas.axes.clear()
        
        if len(self.history['step']) < 2:
            self.canvas.draw()
            return
        
        steps = self.history['step']
        
        # Ana parametreler
        self.canvas.axes.plot(steps, self.history['atp'], 
                             label='ATP', color='gold', linewidth=2)
        self.canvas.axes.plot(steps, self.history['glucose'], 
                             label='Glikoz', color='brown', linewidth=1.5)
        self.canvas.axes.plot(steps, self.history['oxygen'], 
                             label='Oksijen', color='cyan', linewidth=1.5)
        self.canvas.axes.plot(steps, self.history['total_waste'],
                             label='Atık', color='gray', linewidth=1.5)
        
        # İkinci Y ekseni: Boyut
        ax2 = self.canvas.axes.twinx()
        ax2.plot(steps, self.history['size'], 
                label='Boyut', color='green', linewidth=3, linestyle='--')
        ax2.set_ylabel('Hücre Boyutu', color='green', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0.8, max(self.history['size']) * 1.2 if self.history['size'] else 2.0)
        
        self.canvas.axes.set_xlabel('Adım', fontweight='bold')
        self.canvas.axes.set_ylabel('Miktar', fontweight='bold')
        self.canvas.axes.set_title('Hücre Metabolizması', fontweight='bold', fontsize=14)
        self.canvas.axes.legend(loc='upper left')
        ax2.legend(loc='upper right')
        self.canvas.axes.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def reset_simulation(self):
        """Simülasyonu sıfırla"""
        self.timer.stop()
        self.is_running = False
        self.btn_start.setText("Başlat")
        
        self.obs = self.env.reset()
        
        # Geçmişi temizle
        for key in self.history:
            self.history[key] = []
        
        self.update_display()
        print("Simülasyon sıfırlandı!")


def main():
    app = QtWidgets.QApplication(sys.argv)
    # Modern stil
    app.setStyle('Fusion')
    
    window = CellSimulationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()