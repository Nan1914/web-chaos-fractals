import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go # Nueva librería para 3D


# --- Configuración General ---
st.set_page_config(
    page_title="El Laboratorio del CAOS",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)
# CSS para ocultar el menú superior y el footer de "Made with Streamlit"
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 1. BARRA LATERAL (NAVEGACIÓN)
# ==========================================
st.sidebar.title("🌀 Explora el CAOS")

categoria = st.sidebar.radio(
    "Categoría:",
    ["Sistemas Dinámicos", "Fractales", "Cuencas de Atracción"]
)

st.sidebar.markdown("---")

opcion = ""

if categoria == "Sistemas Dinámicos":
    opcion = st.sidebar.selectbox(
        "Experimento:",
        ("Mapa Logístico (2D)", "Atractor de Lorenz (3D)", "Atractor de Thomas (3D)")
    )
    
elif categoria == "Fractales":
    opcion = st.sidebar.selectbox(
        "Experimento:",
        ("Conjunto de Mandelbrot",)
    )

elif categoria == "Cuencas de Atracción":
    opcion = st.sidebar.selectbox(
        "Experimento:",
        ("Oscilador de Duffing", "Fractal de Newton (Próximamente)")
    )

# -- Sección de Referencias --
st.sidebar.markdown("---")
st.sidebar.write("### Referencias")
mostrar_referencias = st.sidebar.checkbox("Ver Bibliografía")


# ==========================================
# 2. CONTENIDO PRINCIPAL
# ==========================================

if mostrar_referencias:
    st.title("📚 Bibliografía y Recursos")
    st.markdown("Recursos esenciales para entender el caos y la complejidad.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Textos Fundamentales")
        st.markdown("""
        * **James Gleick** - *Caos: La creación de una ciencia*.
        * **Steven Strogatz** - *Nonlinear Dynamics and Chaos*.
        * **Benoît Mandelbrot** - *The Fractal Geometry of Nature*.
        """)
    
    with col2:
        st.subheader("Conceptos Clave")
        st.markdown("""
        * **Atractor Extraño:** Un conjunto de puntos hacia donde evoluciona un sistema caótico.
        * **Autosemejanza:** Patrones que se repiten a diferentes escalas.
        * **Efecto Mariposa:** Sensibilidad extrema a las condiciones iniciales.
        """)

else:
    # ---------------------------------------
    # SISTEMAS DINÁMICOS
    # ---------------------------------------
    if opcion == "Mapa Logístico (2D)":
        st.title("El Mapa Logístico")
        st.markdown("""
        ### El caos en las poblaciones
        Popularizado por el biólogo **Robert May** en 1976, este modelo sencillo demuestra cómo un comportamiento complejo puede surgir de reglas deterministas muy simples.
        Originalmente se diseñó para describir la evolución demográfica de una población (como conejos) donde los recursos son limitados.
        """)
        
        st.latex(r"x_{n+1} = r \cdot x_n (1 - x_n)")
        
        st.info("""
        * **$x_n$**: Población actual (entre 0 y 1).
        * **$r$**: Tasa de reproducción.
        * **Caos:** Ocurre cuando $r > 3.57$, donde la población nunca se repite.
        """)
        
        st.divider()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("#### Controles")
            n_iter = st.slider("Iteraciones", 500, 2000, 1000)
            r_range = st.slider("Rango de r", 2.5, 4.0, (2.5, 4.0))
        
        with col2:
            r = np.linspace(r_range[0], r_range[1], 1000)
            x = 1e-5 * np.ones(1000)
            for i in range(100): x = r * x * (1 - x)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            for i in range(n_iter):
                x = r * x * (1 - x)
                ax.scatter(r, x, s=0.1, c='cyan', alpha=0.1)
            
            ax.axis('off')
            st.pyplot(fig)

    elif opcion == "Atractor de Lorenz (3D)":
        st.title("Atractor de Lorenz")
        st.markdown("""
        ### El nacimiento del Efecto Mariposa
        En 1963, el meteorólogo **Edward Lorenz** estaba simulando patrones climáticos en su ordenador. Al redondear unos decimales y volver a correr la simulación, descubrió que el resultado cambiaba drásticamente.
        
        Había descubierto la **sensibilidad a las condiciones iniciales**: el aleteo de una mariposa en Brasil puede provocar un tornado en Texas.
        """)
        
        st.latex(r"""
        \begin{cases}
        \frac{dx}{dt} = \sigma(y-x) \\
        \frac{dy}{dt} = x(\rho-z)-y \\
        \frac{dz}{dt} = xy - \beta z
        \end{cases}
        """)
        
        st.divider()

        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("#### Parámetros")
            sigma = st.slider("Sigma (Prandtl)", 0.0, 20.0, 10.0)
            rho = st.slider("Rho (Rayleigh)", 0.0, 50.0, 28.0)
            beta = st.slider("Beta", 0.0, 5.0, 2.66)
            paleta = st.selectbox("Color", ("Viridis", "Ice", "Plasma", "Turbo"))
        
        with col2:
            dt = 0.01
            num_steps = 10000
            xs, ys, zs = np.empty(num_steps), np.empty(num_steps), np.empty(num_steps)
            xs[0], ys[0], zs[0] = (0.1, 1.0, 1.05)

            for i in range(num_steps - 1):
                xs[i+1] = xs[i] + (sigma * (ys[i] - xs[i])) * dt
                ys[i+1] = ys[i] + (xs[i] * (rho - zs[i]) - ys[i]) * dt
                zs[i+1] = zs[i] + (xs[i] * ys[i] - beta * zs[i]) * dt

            fig = go.Figure(data=go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color=zs, colorscale=paleta, width=2), opacity=0.8
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='#0E1117',
                scene=dict(bgcolor='#0E1117', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
            )
            st.plotly_chart(fig, use_container_width=True)

    elif opcion == "Atractor de Thomas (3D)":
        st.title("Atractor de Thomas")
        st.markdown("""
        ### Simetría Cíclica
        Propuesto por **René Thomas**, este sistema es interesante por su simetría rotacional. A diferencia del de Lorenz, que tiene "dos alas", el atractor de Thomas forma una red compleja similar a una nube de trayectorias.
        
        El parámetro clave es la fricción $b$. Si $b$ es cercano a 0, el sistema es extremadamente caótico y llena todo el espacio. 
        
        P.D.: Con $b=0.15$, aparece una estructura realmente bella.
        """)
        
        st.latex(r"""
        \begin{cases}
        \dot{x} = \sin(y) - b x \\
        \dot{y} = \sin(z) - b y \\
        \dot{z} = \sin(x) - b z
        \end{cases}
        """)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            b = st.slider("Fricción (b)", 0.0, 1.0, 0.205)
            paleta = st.selectbox("Color", ("Viridis", "Ice", "Plasma", "Turbo"))
        
        with col2:
            dt = 0.05
            n_steps = 25000
            xs, ys, zs = np.empty(n_steps), np.empty(n_steps), np.empty(n_steps)
            xs[0], ys[0], zs[0] = (0.1, 0, 0) 

            for i in range(n_steps - 1):
                xs[i+1] = xs[i] + (np.sin(ys[i]) - b * xs[i]) * dt
                ys[i+1] = ys[i] + (np.sin(zs[i]) - b * ys[i]) * dt
                zs[i+1] = zs[i] + (np.sin(xs[i]) - b * zs[i]) * dt

            fig = go.Figure(data=go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines',
                line=dict(color=xs+ys+zs, colorscale=paleta, width=1.5), opacity=0.6
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='#0E1117',
                scene=dict(bgcolor='#0E1117', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------
    # FRACTALES
    # ---------------------------------------
    elif opcion == "Conjunto de Mandelbrot":
        st.title("Conjunto de Mandelbrot")
        st.markdown("""
        ### La huella digital de Dios
        Descubierto por **Benoît Mandelbrot** en 1980, este conjunto es el objeto más famoso de las matemáticas modernas.
        Representa un mapa de todos los conjuntos de Julia posibles. Lo asombroso es su **autosimilitud**: si haces zoom en el borde, encontrarás copias infinitas de la figura original.
        """)
        
        st.latex(r"z_{n+1} = z_n^2 + c")
        st.info("Un punto $c$ pertenece al conjunto si, al iterar la ecuación partiendo de $z=0$, el valor no tiende a infinito.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            resolucion = st.slider("Resolución", 200, 800, 400)
            iteraciones = st.slider("Iteraciones", 20, 100, 50)
        
        with col2:
            def mandelbrot(h, w, max_iter):
                y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
                c = x + y*1j
                z = c
                divtime = max_iter + np.zeros(z.shape, dtype=int)
                for i in range(max_iter):
                    z = z**2 + c
                    diverge = z*np.conj(z) > 2**2            
                    div_now = diverge & (divtime == max_iter)  
                    divtime[div_now] = i                     
                    z[diverge] = 2                           
                return divtime

            with st.spinner('Calculando fractal...'):
                plt.figure(figsize=(10, 10), facecolor='#0E1117')
                fractal = mandelbrot(resolucion, resolucion, iteraciones)
                plt.imshow(fractal, cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
                plt.axis('off')
                st.pyplot(plt)

    # ---------------------------------------
    # CUENCAS DE ATRACCIÓN
    # ---------------------------------------

    elif opcion == "Oscilador de Duffing":
        st.title("Cuencas Fractales de Duffing")
        st.markdown("""
        ### El mapa del Doble Pozo con Forzamiento
        Visualizamos la evolución de un sistema con dos estados estables excitado externamente.
        La ecuación incluye ahora un término de forzamiento periódico $F \cos(\omega t)$.
        
        * **Espirales:** Al añadir fuerza externa y variar la fricción, las fronteras entre las cuencas se vuelven fractales complejos.
        
        * **Colores:** Indican la fase final (ángulo) en el espacio de fases, revelando la compleja estructura espiral de los atractores.
        * **Parámetros del artículo:** $\delta=0.05, F=0.098, \omega=1.15$.
        """)
        
        st.latex(r"\ddot{x} + \delta \dot{x} - x + x^3 = F \cos(\omega t)")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write("#### Parámetros")
            delta = st.slider("Amortiguamiento ($\delta$)", 0.0, 0.5, 0.05, step=0.005, format="%.3f")
            F = st.slider("Fuerza externa (F)", 0.0, 0.5, 0.098, step=0.001, format="%.3f")
            omega = st.slider("Frecuencia ($\omega$)", 0.0, 2.0, 1.15, step=0.01)
            
            st.divider()
            resolucion = st.slider("Resolución (px)", 200, 800, 500)
            t_max = st.slider("Tiempo simulación", 50, 200, 100)
            
            st.info("""
            **Aviso:** Con alta resolución y tiempo largo, el cálculo puede tardar. ¡Paciencia!
            """)

        with col2:
            def duffing_basins_paper_style(res, delta, time_steps, F, omega):
                x = np.linspace(-2.5, 2.5, res)
                y = np.linspace(-2.5, 2.5, res)
                X, Y = np.meshgrid(x, y)
                
                dt = 0.05
                steps = int(time_steps / dt)
                t = 0.0 
                
                # Método Euler-Cromer
                for _ in range(steps):
                    Y_new = Y + (X - X**3 - delta * Y + F * np.cos(omega * t)) * dt
                    X_new = X + Y_new * dt
                    
                    X, Y = X_new, Y_new
                    
                    mask = (X**2 + Y**2 < 50) 
                    X[~mask] = np.nan
                    Y[~mask] = np.nan
                    
                    t += dt 
                
                basins_angle = np.arctan2(Y, X)
                return basins_angle

            with st.spinner('Calculando la estructura fractal...'):
                # Color de fondo de la figura global
                fig = plt.figure(figsize=(10, 10), facecolor='#0E1117')
                
                basins = duffing_basins_paper_style(resolucion, delta, t_max, F, omega)
                
                plt.imshow(basins, cmap='turbo', extent=[-2.5, 2.5, -2.5, 2.5], origin='lower')
                
                plt.title(f"Duffing Fractal ($\delta={delta:.2f}, F={F:.3f}, \omega={omega:.2f}$)", color='white')
                plt.xlabel('$x$', color='white', fontsize=14)
                plt.ylabel('$\dot{x}$', color='white', fontsize=14)
                
                # --- CORRECCIÓN DEL BORDE BLANCO ---
                ax = plt.gca()
                # 1. Forzar el color de fondo de los ejes
                ax.set_facecolor('#0E1117')
                
                # Configuración de colores para ejes y texto
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values(): spine.set_color('white')
                
                # 2. Eliminar márgenes extra
                plt.tight_layout()

                # Pasar la figura explícitamente ayuda a veces
                st.pyplot(fig)
 
    elif opcion == "Fractal de Newton (Próximamente)":
        st.title("Fractal de Newton")
        st.markdown("""
        ### Método de Newton-Raphson
        Este fractal surge al intentar encontrar las raíces de un polinomio (soluciones donde la ecuación vale cero) usando un método numérico iterativo.
        
        Dependiendo de dónde empieces en el plano complejo, el punto acabará convergiendo a una raíz u otra. El "mapa" de qué punto va a qué raíz crea fronteras fractales increíblemente bellas.
        """)
        st.latex(r"z_{n+1} = z_n - \frac{f(z_n)}{f'(z_n)}")
        st.info("🚧 Sección en construcción.")











