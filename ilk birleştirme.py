import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point, MultiPolygon
from matplotlib.widgets import Button, TextBox
from scipy.spatial import Voronoi
import copy

# --- SİMÜLASYON SABİTLERİ ---
ITERASYON_SAYISI = 80
ANIMASYON_HIZI_MS = 100
MIN_KOSSE_SAYISI = 3 
MIN_DRONE_SAYISI = 1 
MAX_DRONE_SAYISI = 200 
INITIAL_DRONE_COUNT = 40

class Environment:
    def __init__(self, boundary_coords):
        self.boundary_coords = np.array(boundary_coords)
        self.polygon = Polygon(self.boundary_coords)
        
        centroid_obj = self.polygon.centroid
        self.centroid = np.array([centroid_obj.x, centroid_obj.y])
        
        min_x, min_y, max_x, max_y = self.polygon.bounds
        genislik = max_x - min_x
        yukseklik = max_y - min_y
        uzaklik = max(genislik, yukseklik) * 100 

        self.ghost_points = np.array([
            [min_x - uzaklik, min_y - uzaklik],
            [max_x + uzaklik, min_y - uzaklik],
            [max_x + uzaklik, max_y + uzaklik],
            [min_x - uzaklik, max_y + uzaklik]
        ])

class BaseSolver:
    def __init__(self, env, drone_count, initial_positions, colors):
        self.env = env
        self.drone_count = drone_count
        self.positions = copy.deepcopy(initial_positions)
        self.colors = colors

    def step(self):
        raise NotImplementedError

    def draw(self, ax, title):
        ax.clear()
        
        min_x, min_y, max_x, max_y = self.env.polygon.bounds
        margin_x = (max_x - min_x) * 0.1
        margin_y = (max_y - min_y) * 0.1
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)
        
        ax.plot(*self.env.polygon.exterior.xy, color='black', linewidth=3, zorder=5)
        
        tum_noktalar = np.vstack((self.positions, self.env.ghost_points))
        vor = Voronoi(tum_noktalar)
        
        for i in range(self.drone_count):
            point_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[point_idx]
            
            if -1 in region_vertices_indices or not region_vertices_indices: continue
            
            hucre_kose_noktalari = [vor.vertices[v] for v in region_vertices_indices]
            
            if len(hucre_kose_noktalari) < 3: continue
                
            try:
                hucre_poligonu = Polygon(hucre_kose_noktalari).buffer(0)
                kirpilmis_hucre = self.env.polygon.intersection(hucre_poligonu)

                if not kirpilmis_hucre.is_empty:
                    renk = self.colors[i]
                    if isinstance(kirpilmis_hucre, MultiPolygon):
                        for poly in kirpilmis_hucre.geoms:
                            ax.fill(*poly.exterior.xy, color=renk, alpha=0.7, zorder=0, edgecolor='gray', linewidth=0.5)
                    elif isinstance(kirpilmis_hucre, Polygon):
                        ax.fill(*kirpilmis_hucre.exterior.xy, color=renk, alpha=0.7, zorder=0, edgecolor='gray', linewidth=0.5)
            except Exception:
                pass 

        ax.scatter(self.positions[:, 0], self.positions[:, 1], c=self.colors, edgecolor='white', s=80, zorder=10)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)

class APFSolver(BaseSolver):
    def __init__(self, env, drone_count, initial_positions, colors):
        super().__init__(env, drone_count, initial_positions, colors)
        self.velocities = np.zeros((self.drone_count, 2))
        
        self.DT = 0.04
        self.DAMPING = 0.6          
        self.K_NOISE = 0.01
        self.K_ATTRACT = 0.25       
        self.K_REPEL_AGENT = 0.12   
        self.REPULSION_DIST = 0.35 
        self.K_REPEL_BOUND = 0.4 
        self.BOUND_MARGIN = 0.15

    def step(self):
        forces = np.zeros((self.drone_count, 2))
        
        for i in range(self.drone_count):
            pos_i = self.positions[i]
            p_point = Point(pos_i)
            
            vec_to_center = self.env.centroid - pos_i
            forces[i] += self.K_ATTRACT * vec_to_center
            
            for j in range(self.drone_count):
                if i == j: continue
                vec_ij = pos_i - self.positions[j]
                dist_ij = np.linalg.norm(vec_ij)
                if dist_ij < self.REPULSION_DIST:
                    magnitude = self.K_REPEL_AGENT * (1.0 / (max(dist_ij, 0.04)))
                    forces[i] += (vec_ij / (dist_ij + 1e-6)) * magnitude

            dist_to_boundary = p_point.distance(self.env.polygon.exterior)
            is_inside = self.env.polygon.contains(p_point)
            
            if is_inside and dist_to_boundary < self.BOUND_MARGIN:
                closest_p = self.env.polygon.exterior.interpolate(self.env.polygon.exterior.project(p_point))
                vec_from_boundary = pos_i - np.array(closest_p.coords[0])
                dist = np.linalg.norm(vec_from_boundary)
                magnitude = self.K_REPEL_BOUND * (1.0 / (dist_to_boundary + 0.01))
                forces[i] += (vec_from_boundary / (dist + 1e-6)) * magnitude
            elif not is_inside:
                forces[i] += (self.env.centroid - pos_i) * 5.0

        forces += (np.random.rand(self.drone_count, 2) - 0.5) * self.K_NOISE
        self.velocities = (self.velocities + forces * self.DT) * self.DAMPING
        self.positions += self.velocities * self.DT
        
        for i in range(self.drone_count):
            p_point = Point(self.positions[i])
            if not self.env.polygon.contains(p_point):
                closest = self.env.polygon.exterior.interpolate(self.env.polygon.exterior.project(p_point))
                self.positions[i] = np.array(closest.coords[0])
                self.velocities[i] = np.zeros(2)

class LloydSolver(BaseSolver):
    def step(self):
        tum_noktalar = np.vstack((self.positions, self.env.ghost_points))
        vor = Voronoi(tum_noktalar)
        yeni_konumlar = []
        
        for i in range(self.drone_count): 
            point_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[point_idx]
            
            if -1 in region_vertices_indices or not region_vertices_indices:
                yeni_konumlar.append(self.positions[i])
                continue
                
            hucre_kose_noktalari = [vor.vertices[v] for v in region_vertices_indices]
            if len(hucre_kose_noktalari) < 3:
                yeni_konumlar.append(self.positions[i])
                continue
                
            try:
                hucre_poligonu = Polygon(hucre_kose_noktalari).buffer(0)
                kirpilmis_hucre = self.env.polygon.intersection(hucre_poligonu)
                
                if kirpilmis_hucre.area > 0:
                    sentroid = kirpilmis_hucre.centroid
                    yeni_konumlar.append(list(sentroid.coords)[0])
                else:
                    yeni_konumlar.append(self.positions[i])
            except Exception:
                yeni_konumlar.append(self.positions[i])
                
        self.positions = np.array(yeni_konumlar)

class SimulationManager:
    def __init__(self):
        self.drone_count = INITIAL_DRONE_COUNT
        self.boundary_coords = None
        self.colors = None
        self.ani = None 
        
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('Otonom Dağılım Simülasyonu')
        
        manager = plt.get_current_fig_manager()
        try:
            manager.window.state('zoomed') 
        except Exception:
            try:
                manager.full_screen_toggle() 
            except Exception:
                pass
        
    def setup_ui(self):
        self.ax_setup = self.fig.add_subplot(111)
        self.ax_setup.set_title("ADIM 1: Aşağıdan Drone Sayısını Girin ve 'Alanı Çiz' Butonuna Basın", fontsize=16)
        
        self.ax_setup.set_aspect('equal', adjustable='box')
        self.ax_setup.set_xlim(0, 1)
        self.ax_setup.set_ylim(0, 1)
        self.ax_setup.grid(True, linestyle=':', alpha=0.6)
        
        # İstediğin gibi left'i 0'a yakın (0.05) yapıp butonlara yer açıyoruz
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.9) 
        
        # Butonları tam ortaya, grafiğin hemen altına hizaladık
        ax_textbox = self.fig.add_axes([0.35, 0.08, 0.1, 0.06]) 
        self.text_box = TextBox(ax_textbox, 'Drone Sayısı: ', initial=str(self.drone_count))
        
        ax_button = self.fig.add_axes([0.48, 0.08, 0.2, 0.06]) 
        self.alan_tanimla_button = Button(ax_button, 'ADIM 2: Alanı Çizmeye Başla', color='lightgreen')

        self.fig.canvas.draw_idle()

        def on_start_clicked(event):
            try:
                val = int(self.text_box.text)
                if MIN_DRONE_SAYISI <= val <= MAX_DRONE_SAYISI:
                    self.drone_count = val
                else:
                    self.ax_setup.set_title(f"HATA: Lütfen {MIN_DRONE_SAYISI} ile {MAX_DRONE_SAYISI} arası bir değer girin!", color='red')
                    self.fig.canvas.draw_idle()
                    return
            except ValueError:
                self.ax_setup.set_title("HATA: Lütfen geçerli bir tam sayı girin!", color='red')
                self.fig.canvas.draw_idle()
                return

            self.text_box.set_active(False)
            self.alan_tanimla_button.set_active(False)
            ax_textbox.set_visible(False)
            ax_button.set_visible(False)

            self.ax_setup.set_title(f"ADIM 3: {self.drone_count} Drone İçin Alan Köşelerini Tıklayın (Bitirmek için ENTER)", fontsize=16, color='red')
            self.fig.canvas.draw_idle()

            try:
                koseler = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
            except Exception:
                return

            if len(koseler) >= MIN_KOSSE_SAYISI:
                self.boundary_coords = np.array(koseler)
                self.colors = np.random.rand(self.drone_count, 3) 
                
                self.start_simulation()
            else:
                self.ax_setup.set_title("HATA: Yeterli köşe seçilmedi! Programı yeniden başlatın.", color='red')
                self.fig.canvas.draw_idle()

        self.alan_tanimla_button.on_clicked(on_start_clicked)
        plt.show() 

    def start_simulation(self):
        self.fig.clf() 
        
        # Simülasyon ekranı için boşlukları tam ekrana yayıyoruz
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.92, wspace=0.1)
        
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        env = Environment(self.boundary_coords)
        initial_positions = np.array([env.centroid] * self.drone_count, dtype=float)
        gurultu_miktari = 0.01 if max(env.polygon.bounds) <= 1.0 else 1.0
        initial_positions += (np.random.rand(self.drone_count, 2) - 0.5) * gurultu_miktari
        
        lloyd_solver = LloydSolver(env, self.drone_count, initial_positions, self.colors)
        apf_solver = APFSolver(env, self.drone_count, initial_positions, self.colors)

        def update(frame):
            lloyd_solver.step()
            lloyd_solver.draw(self.ax1, f"Voronoi (Lloyd)\nİterasyon: {frame + 1} / {ITERASYON_SAYISI}")
            
            apf_solver.step()
            apf_solver.draw(self.ax2, f"Potansiyel Alan (APF)\nİterasyon: {frame + 1} / {ITERASYON_SAYISI}")

        self.ani = FuncAnimation(self.fig, update, frames=ITERASYON_SAYISI, interval=ANIMASYON_HIZI_MS, repeat=False)
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    app = SimulationManager()
    app.setup_ui()