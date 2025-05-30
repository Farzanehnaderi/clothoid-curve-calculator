import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from math import sqrt, sin, cos, tan, radians, degrees, atan2, ceil, floor
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image, ImageTk
import sv_ttk

class ClothoidApp:
    
    def calculate_and_plot(self):
        try:
            L_s = float(self.entry_Ls.get())
            R_c = float(self.entry_Rc.get())
            delta_deg = float(self.entry_delta.get())
            max_L_segment = float(self.entry_max_L_segment.get())
            Km_T = float(self.entry_Km_T.get())
            l_or_r = int(self.direction.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")
            return

        delta = radians(delta_deg)
        A_s = sqrt(L_s * R_c)
        teta_s = L_s / (2 * R_c)
        delta_c = delta_deg - 2 * degrees(teta_s)
        L_c = R_c * radians(delta_c) 
        X_s = L_s
        Y_s = (L_s**3) / (6 * A_s**2)
        k = X_s - R_c * sin(teta_s)
        p = Y_s - R_c * (1 - cos(teta_s))
        E_s = (R_c + p) / cos(radians(delta_c / 2)) - R_c
        T_s = k + (R_c + p) * tan(radians(delta_deg / 2))
        Km_A = Km_T - T_s

        L1 = (floor(Km_A / max_L_segment) + 1) * max_L_segment - Km_A
        rem = L_s - L1
        n = floor(rem / max_L_segment)
        L_segments = [L1] + [max_L_segment]*n
        r = rem % max_L_segment
        if r > 1e-6:
            L_segments.append(r)
        tool_caman = np.cumsum([0] + L_segments)
        x_i = tool_caman
        y_i = (x_i**3) / (6 * A_s**2)

        x_o = X_s - R_c * sin(teta_s)
        y_o = p + R_c
        r2 = max_L_segment - r
        L_segments2 = [r2]
        n2 = floor((R_c * radians(delta_c) - r2) / max_L_segment)
        L_segments2 += [max_L_segment]*n2
        r3 = (R_c * radians(delta_c) - r2) % max_L_segment
        if r3 > 1e-6:
            L_segments2.append(r3)
        tool_caman2 = np.cumsum([0] + L_segments2)
        gis = degrees(atan2((x_i[-1] - x_o), (y_i[-1] - y_o)))
        gis_ps = gis - np.degrees(tool_caman2 / R_c)
        x_cur = x_o + R_c * np.sin(np.radians(gis_ps))
        y_cur = y_o + R_c * np.cos(np.radians(gis_ps))

        x_end_clo2 = T_s + T_s * sin(radians(90 - delta_deg))
        y_end_clo2 = T_s * cos(radians(90 - delta_deg))
        R_mat = np.array([[cos(radians(-delta_deg)), -sin(radians(-delta_deg))],
                          [sin(radians(-delta_deg)),  cos(radians(-delta_deg))]])
        L_segments3 = [max_L_segment - r3]
        n3 = floor((L_s - L_segments3[0]) / max_L_segment)
        L_segments3 += [max_L_segment]*n3
        r4 = (L_s - L_segments3[0]) % max_L_segment
        if r4 > 1e-6:
            L_segments3.append(r4)
        tool_caman3 = np.cumsum([0] + L_segments3)
        x_i_test = tool_caman  
        y_i_test = (x_i_test**3) / (6*A_s**2)
        test2 = R_mat @ np.vstack([x_i_test, y_i_test])
        x_p_2 = x_end_clo2 + (-test2[0, ::-1])
        y_p_2 = y_end_clo2 + test2[1, ::-1]
        x_p = np.concatenate([x_i, x_cur[1:], x_p_2[1:]])
        y_p = np.concatenate([y_i, y_cur[1:], y_p_2[1:]])

        if l_or_r == 0:
            x_p = -x_p
            x_i = -x_i
            x_cur = -x_cur
            x_p_2 = -x_p_2
            x_o = -x_o
            x_end_clo2 = -x_end_clo2
            X_s = -X_s
            
        tool_caman_all = np.concatenate([L_segments, L_segments2, L_segments])
        tool_caman_all = np.insert(tool_caman_all, 0, 0)
        Km_points = Km_A + np.cumsum(tool_caman_all)
        #l_p = np.sqrt(x_p**2 + y_p**2)
        l_p = np.linalg.norm(np.diff(np.column_stack([x_p, y_p]), axis=0), axis=1)
        l_p = np.insert(l_p, 0, 0) 
        dx = np.diff(x_p)
        dy = np.diff(y_p)
        angles_rad = np.arctan2(dy, dx)
        angles_deg = np.degrees(angles_rad)

        z_p = np.insert(np.cumsum(np.diff(np.unwrap(np.radians(angles_deg)))), 0, 0)
        z_p = np.degrees(z_p)

        min_len = min(len(tool_caman_all), len(z_p), len(l_p), len(Km_points))
        self.construction_data = {
    "Point": list(range(min_len)),
    "Arc_Length": tool_caman_all[:min_len],
    "Deflection_Angle": z_p[:min_len],
    "Chord": l_p[:min_len],
    "Kilometrage": Km_points[:min_len]
}

        self.calculated_params = {
            "Clothoid Parameter (A)": A_s,
            "Theta_s (radians)": teta_s,
            "Theta_s (degrees)": degrees(teta_s),
            "Delta_c (degrees)": delta_c,
            "k": k,
            "p": p,
            "External Distance (E_s)": E_s,
            "Tangent Distance (T_s)": T_s,
            "X_s": X_s,
            "Y_s": Y_s,
            "Length of Circular Arc (L_c)": L_c,
            "Total Curve Length": 2 * L_s + L_c}

        self.table_text.delete(1.0, tk.END)
        header = "Point\tArc Length\tDeflection Angle\tChord\tKilometrage\n"
        self.table_text.insert(tk.END, header)
        min_len = min(len(tool_caman_all), len(z_p), len(l_p), len(Km_points))
        for i in range(min_len):
            self.table_text.insert(tk.END, 
                f"{i}\t{tool_caman_all[i]:.3f}\t{z_p[i]:.3f}\t{l_p[i]:.3f}\t{Km_points[i]:.3f}\n")

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#F8F8F8')
        ax.plot(x_i, y_i, color='#4E79A7', linewidth=2.5, label='Clothoid 1')
        ax.plot(x_cur, y_cur, color='#59A14F', linewidth=2.5, label='Circular Arc')
        ax.plot(x_p_2, y_p_2, color='#E15759', linewidth=2.5, label='Clothoid 2')
        ax.plot(x_p, y_p, 'o', color='#F28E2B', markersize=4, label='Points')
        ax.plot(x_o, y_o, 'ko', markersize=6, label='Center')

        for i, (xi, yi) in enumerate(zip(x_p, y_p)):
            ax.annotate(str(i), (xi, yi), textcoords="offset points", 
                        xytext=(0, 4), ha='center', fontsize=8)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Clothoid Curve", fontsize=12)
        ax.axis("equal")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        if l_or_r == 0:
         ax.plot([0, -T_s], [0, 0], 'k--', label="Tangent In")
         ax.plot([-T_s, x_end_clo2], [0, y_end_clo2], 'k--', label="Tangent Out")
        else:
         ax.plot([0, T_s], [0, 0], 'k--', label="Tangent In")
         ax.plot([T_s, x_end_clo2], [0, y_end_clo2], 'k--', label="Tangent Out")
    
        if l_or_r == 0:
         ax.plot([-T_s, x_o], [0, y_o], 'k-.', label="Bisector")
        else:
         ax.plot([T_s, x_o], [0, y_o], 'k-.', label="Bisector")

        ax.plot([x_o, X_s], [y_o, Y_s], 'c--', label="To Circular Start")
        ax.plot([x_cur[-1], x_o], [y_cur[-1], y_o], 'c--', label="To Circular End")
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(tk.END, "Calculated Parameters:\n\n")
        for param, value in self.calculated_params.items():
            self.params_text.insert(tk.END, f"{param:<25}: {value:.6f}\n")
        self.params_text.tag_configure("title", foreground=self.primary_color, 
                                       font=('Consolas', 10, 'bold'))
        self.params_text.tag_add("title", "1.0", "1.19")
        self.canvas.draw()

    def __init__(self, root):
        self.root = root
        self.root.title("Clothoid Curve Calculator")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        sv_ttk.set_theme("light") 
        self.style = ttk.Style()
        self.primary_color = "#4F6D7A"
        self.secondary_color = "#E8DAB2"
        self.accent_color = "#DD6E42"
        self.background_color = "#F0F4F8"
        self.text_color = "#2E2E2E"
        self.style.configure('TFrame', background=self.background_color)
        self.style.configure('TLabel', background=self.background_color, 
                           font=('Segoe UI', 10), foreground=self.text_color)
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'),
                           background=self.accent_color, foreground='white')
        self.style.configure('TEntry', font=('Segoe UI', 10), 
                           fieldbackground='white')
        self.style.configure('TNotebook', background=self.background_color)
        self.style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'), 
                           padding=[15, 5], background=self.secondary_color)
        self.create_welcome_page()
        
    def create_welcome_page(self):
        self.welcome_frame = ttk.Frame(self.root)
        self.welcome_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(self.welcome_frame)
        header_frame.pack(pady=(50, 20))

        try:
            logo_img = Image.open("logo_tehran.jpg").resize((80, 80))
            self.logo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(header_frame, image=self.logo)
            logo_label.pack(side=tk.LEFT, padx=10)
        except:
            pass
        
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT)
        
        ttk.Label(title_frame, text="Clothoid Curve Calculator", 
                 font=('Segoe UI', 24, 'bold'), 
                 foreground=self.primary_color).pack(anchor=tk.W)
        ttk.Label(title_frame, text="Transition Curve Design Tool", 
                 font=('Segoe UI', 14), 
                 foreground=self.text_color).pack(anchor=tk.W)
        
        info_frame = ttk.Frame(self.welcome_frame)
        info_frame.pack(pady=20)
        
        info_text = """
        University: University of Tehran
        Professor: Dr. Yousef Kanani
        Developer: Farzaneh Naderi
        Project: Clothoid Transition Curve Design
        """
        
        ttk.Label(info_frame, text=info_text.strip(), 
                 font=('Segoe UI', 12), 
                 justify=tk.LEFT).pack()
        
        btn_frame = ttk.Frame(self.welcome_frame)
        btn_frame.pack(pady=40)
        
        help_btn = ttk.Button(btn_frame, text="HELP", command=self.show_help,
                            style='TButton')
        help_btn.pack(pady=10, ipadx=20, ipady=8)
        
        start_btn = ttk.Button(btn_frame, text="Start Program", command=self.create_main_app,
                             style='TButton')
        start_btn.pack(pady=10, ipadx=20, ipady=8)
        
        footer_frame = ttk.Frame(self.welcome_frame)
        footer_frame.pack(side=tk.BOTTOM, pady=20)
        
        ttk.Label(footer_frame, text="© 2025 Civil Engineering Department - University of Tehran",
                 font=('Segoe UI', 9), foreground='#666666').pack()
    
    def show_help(self):
        help_text = """Clothoid Curve Calculator Help:
        
A clothoid (or Euler spiral) is a curve whose curvature changes linearly with its length. 
It's commonly used in road and railway design as a transition curve between straight 
sections and circular curves.

This program calculates and visualizes a clothoid curve based on:
- L_s: Length of the clothoid
- R_c: Radius of the circular curve
- delta: Deflection angle
- Segment length: For construction points
- Direction: Right or left turn

The program provides:
1. Input parameters tab
2. Calculated parameters tab
3. Construction table tab
4. Graphical visualization tab
"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x400")
        help_window.resizable(False, False)
        
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        text = tk.Text(text_frame, wrap=tk.WORD, font=('Segoe UI', 11), 
                      padx=10, pady=10, bg='white')
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True)
        
        close_btn = ttk.Button(help_window, text="Close", command=help_window.destroy)
        close_btn.pack(pady=10)
    
    def create_main_app(self):
        self.welcome_frame.destroy()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab_input = ttk.Frame(self.notebook)
        self.tab_params = ttk.Frame(self.notebook)
        self.tab_table = ttk.Frame(self.notebook)
        self.tab_plot = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_input, text="Input Parameters")
        self.notebook.add(self.tab_params, text="Calculated Parameters")
        self.notebook.add(self.tab_table, text="Construction Table")
        self.notebook.add(self.tab_plot, text="Graphical Visualization")
        
        self.create_input_tab()
        self.create_params_tab()
        self.create_table_tab()
        self.create_plot_tab()

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=10)
        
        export_frame = ttk.Frame(btn_frame)
        export_frame.pack(pady=5)
        
        excel_btn = ttk.Button(export_frame, text="Export to Excel", command=self.export_to_excel,
                              style='Accent.TButton')
        excel_btn.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=5)
        
        pdf_btn = ttk.Button(export_frame, text="Export to PDF", command=self.export_to_pdf,
                           style='Accent.TButton')
        pdf_btn.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=5)
        
        self.style.configure('Accent.TButton', background=self.accent_color,
                           foreground='white', font=('Segoe UI', 10, 'bold'))
    
    def create_input_tab(self):
        main_frame = ttk.Frame(self.tab_input)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        input_frame = ttk.Frame(main_frame, style='Card.TFrame')
        input_frame.pack(pady=20, padx=20, fill=tk.X)
        self.style.configure('Card.TFrame', background='white', 
                            relief='solid', borderwidth=1, 
                            bordercolor='#e0e0e0', padding=10)
        
        ttk.Label(input_frame, text="Enter Clothoid Parameters", 
                 font=('Segoe UI', 12, 'bold'), 
                 foreground=self.primary_color).grid(row=0, column=0, 
                                                   columnspan=2, 
                                                   pady=(0, 15), sticky=tk.W)
        
        params = [
            ("Clothoid Length (L_s):", "120", "meters"),
            ("Main Circle Radius (R_c):", "500", "meters"),
            ("Deflection Angle (delta):", "45", "degrees"),
            ("Max Segment Length:", "50", "meters"),
            ("Kilometrage of Intersection (Km_T):", "1000", "km")
        ]
        
        self.entries = []
        for i, (label, default, unit) in enumerate(params):
            row = i + 1  
            
            ttk.Label(input_frame, text=label, 
                     font=('Segoe UI', 10)).grid(row=row, column=0, 
                                               sticky=tk.W, padx=5, pady=8)
            
            entry_frame = ttk.Frame(input_frame)
            entry_frame.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=8)
            
            entry = ttk.Entry(entry_frame, width=15, font=('Segoe UI', 10))
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            
            ttk.Label(entry_frame, text=unit, 
                     font=('Segoe UI', 9), 
                     foreground='#666666').pack(side=tk.LEFT, padx=5)
            
            self.entries.append(entry)
        
        self.entry_Ls, self.entry_Rc, self.entry_delta, self.entry_max_L_segment, self.entry_Km_T = self.entries
        
        direction_frame = ttk.Frame(input_frame)
        direction_frame.grid(row=len(params)+1, column=0, 
                           columnspan=2, pady=(15, 5), sticky=tk.W)
        
        ttk.Label(direction_frame, text="Direction:", 
                 font=('Segoe UI', 10)).pack(side=tk.LEFT)
        
        self.direction = tk.StringVar(value="1")
        
        right_btn = ttk.Radiobutton(direction_frame, text="Right Turn", 
                                   variable=self.direction, value="1",
                                   style='Toolbutton')
        right_btn.pack(side=tk.LEFT, padx=5)
        
        left_btn = ttk.Radiobutton(direction_frame, text="Left Turn", 
                                  variable=self.direction, value="0",
                                  style='Toolbutton')
        left_btn.pack(side=tk.LEFT, padx=5)
        
        self.style.configure('Toolbutton', font=('Segoe UI', 9))

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        calc_btn = ttk.Button(btn_frame, text="Calculate & Plot", 
                             command=self.calculate_and_plot,
                             style='Accent.TButton')
        calc_btn.pack(ipadx=20, ipady=8)
    
    def create_params_tab(self):
        main_frame = ttk.Frame(self.tab_params)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Calculated Parameters", 
                 font=('Segoe UI', 12, 'bold'), 
                 foreground=self.primary_color).pack(anchor=tk.W, pady=(0, 10))
        
        text_frame = ttk.Frame(main_frame, style='Card.TFrame')
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.params_text = tk.Text(text_frame, height=20, width=80, 
                                 font=("Consolas", 10), 
                                 bg='white', fg=self.text_color,
                                 padx=10, pady=10)
        self.params_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.params_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.params_text.config(yscrollcommand=scrollbar.set)
    
    def create_table_tab(self):
        main_frame = ttk.Frame(self.tab_table)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Construction Points Table", 
                 font=('Segoe UI', 12, 'bold'), 
                 foreground=self.primary_color).pack(anchor=tk.W, pady=(0, 10))
        
        text_frame = ttk.Frame(main_frame, style='Card.TFrame')
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.table_text = tk.Text(text_frame, height=25, width=120, 
                                font=("Consolas", 9), 
                                bg='white', fg=self.text_color,
                                padx=10, pady=10)
        
        export_frame = ttk.Frame(self.tab_table)
        export_frame.pack(pady=10)

        excel_btn = ttk.Button(export_frame, text="Export to Excel", command=self.export_to_excel, style='Accent.TButton')
        excel_btn.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=5)

        pdf_btn = ttk.Button(export_frame, text="Export to PDF", command=self.export_to_pdf, style='Accent.TButton')
        pdf_btn.pack(side=tk.LEFT, padx=10, ipadx=15, ipady=5)
    
        self.table_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.table_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table_text.config(yscrollcommand=scrollbar.set)
    
    def create_plot_tab(self):
        main_frame = ttk.Frame(self.tab_plot)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Clothoid Curve Visualization", 
                 font=('Segoe UI', 12, 'bold'), 
                 foreground=self.primary_color).pack(anchor=tk.W, pady=(0, 10))
        
        plot_frame = ttk.Frame(main_frame, style='Card.TFrame')
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig = plt.Figure(figsize=(8, 7), dpi=100, facecolor='none')
        self.fig.patch.set_alpha(0.0)  
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
    
    def export_to_excel(self):
        if not hasattr(self, 'construction_data'):
            messagebox.showerror("Error", "No data to export. Please calculate first.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save Excel File"
            )
            
            if file_path:
                df = pd.DataFrame(self.construction_data)
                df.to_excel(file_path, index=False)
                
                success_window = tk.Toplevel(self.root)
                success_window.title("Success")
                success_window.geometry("300x150")
                
                ttk.Label(success_window, text="Export Successful!", 
                         font=('Segoe UI', 12, 'bold'),
                         foreground=self.primary_color).pack(pady=20)
                
                ttk.Label(success_window, text=f"File saved to:\n{file_path}",
                         font=('Segoe UI', 9)).pack()
                
                ttk.Button(success_window, text="OK", 
                          command=success_window.destroy,
                          style='Accent.TButton').pack(pady=10)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to Excel:\n{str(e)}")
    
    def export_to_pdf(self):
        if not hasattr(self, 'construction_data') or not hasattr(self, 'calculated_params'):
            messagebox.showerror("Error", "No data to export. Please calculate first.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save PDF File"
            )
            
            if file_path:
                pdf = canvas.Canvas(file_path, pagesize=letter)
                width, height = letter
                
                pdf.setFont("Helvetica-Bold", 18)
                pdf.setFillColor(colors.HexColor(self.primary_color))
                pdf.drawString(100, height-50, "Clothoid Transition Curve Report")
                
                try:
                    pdf.drawImage("logo.png", width-150, height-70, 
                                width=50, height=50, mask='auto')
                except:
                    pass
                
                pdf.setFont("Helvetica", 10)
                pdf.setFillColor(colors.black)
                pdf.drawString(100, height-80, "University of Tehran - Civil Engineering Department")
                
                pdf.setFont("Helvetica-Bold", 14)
                pdf.setFillColor(colors.HexColor(self.primary_color))
                pdf.drawString(100, height-120, "Input Parameters:")
                pdf.setFont("Helvetica", 10)
                pdf.setFillColor(colors.black)
                
                y_pos = height-140
                params = [
                    ("Clothoid Length (L_s):", self.entry_Ls.get() + " meters"),
                    ("Main Circle Radius (R_c):", self.entry_Rc.get() + " meters"),
                    ("Deflection Angle (delta):", self.entry_delta.get() + " degrees"),
                    ("Max Segment Length:", self.entry_max_L_segment.get() + " meters"),
                    ("Kilometrage of Intersection (Km_T):", self.entry_Km_T.get() + " km"),
                    ("Direction:", "Right Turn" if self.direction.get() == "1" else "Left Turn")
                ]
                
                for param, value in params:
                    pdf.drawString(120, y_pos, f"{param:<30} {value}")
                    y_pos -= 20
                    if y_pos < 100:
                        pdf.showPage()
                        y_pos = height-50
                        pdf.setFont("Helvetica", 10)
                
                pdf.setFont("Helvetica-Bold", 14)
                pdf.setFillColor(colors.HexColor(self.primary_color))
                pdf.drawString(100, y_pos-20, "Calculated Parameters:")
                pdf.setFont("Helvetica", 10)
                pdf.setFillColor(colors.black)
                y_pos -= 40
                
                for param, value in self.calculated_params.items():
                    pdf.drawString(120, y_pos, f"{param:<30} {value:.6f}")
                    y_pos -= 20
                    if y_pos < 100:
                        pdf.showPage()
                        y_pos = height-50
                        pdf.setFont("Helvetica", 10)
                
                pdf.setFont("Helvetica-Bold", 14)
                pdf.setFillColor(colors.HexColor(self.primary_color))
                pdf.drawString(100, y_pos-20, "Construction Points Table:")
                pdf.setFont("Helvetica", 8)
                y_pos -= 40
                
                data = [["Point", "Arc Length", "Deflection Angle", "Chord", "Kilometrage"]]
                min_len = min(len(self.construction_data["Point"]), 
                             len(self.construction_data["Arc_Length"]),
                             len(self.construction_data["Deflection_Angle"]),
                             len(self.construction_data["Chord"]),
                             len(self.construction_data["Kilometrage"]))
                
                for i in range(min_len):
                    data.append([
                        str(self.construction_data["Point"][i]),
                        f"{self.construction_data['Arc_Length'][i]:.3f}",
                        f"{self.construction_data['Deflection_Angle'][i]:.3f}",
                        f"{self.construction_data['Chord'][i]:.3f}",
                        f"{self.construction_data['Kilometrage'][i]:.3f}"
                    ])
                
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.primary_color)),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 10),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor(self.secondary_color)),
                    ('GRID', (0,0), (-1,-1), 1, colors.HexColor(self.primary_color))
                ]))
                
                table_height = 20 * len(data)
                if y_pos - table_height < 50:  
                 pdf.showPage()
                 y_pos = height - 50  
                 pdf.setFont("Helvetica-Bold", 14)
                 pdf.setFillColor(colors.HexColor(self.primary_color))
                 pdf.drawString(100, y_pos, "Construction Points Table:")
                 y_pos -= 30
                 pdf.setFont("Helvetica", 8)

                table.wrapOn(pdf, width - 200, height)
                table.drawOn(pdf, 100, y_pos - table_height)

                plot_path = "temp_plot.png"
                self.fig.savefig(plot_path, dpi=150, bbox_inches='tight', 
                               facecolor=self.fig.get_facecolor())
                
                pdf.showPage()
                pdf.setFont("Helvetica-Bold", 14)
                pdf.setFillColor(colors.HexColor(self.primary_color))
                pdf.drawString(100, height-50, "Clothoid Curve Plot:")
                pdf.drawImage(plot_path, 50, height-450, width=500, height=350, 
                            preserveAspectRatio=True)
                
                pdf.save()
                
                success_window = tk.Toplevel(self.root)
                success_window.title("Success")
                success_window.geometry("300x150")
                
                ttk.Label(success_window, text="PDF Export Successful!", 
                         font=('Segoe UI', 12, 'bold'),
                         foreground=self.primary_color).pack(pady=20)
                
                ttk.Label(success_window, text=f"File saved to:\n{file_path}",
                         font=('Segoe UI', 9)).pack()
                
                ttk.Button(success_window, text="OK", 
                          command=success_window.destroy,
                          style='Accent.TButton').pack(pady=10)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ClothoidApp(root)
    root.mainloop()
