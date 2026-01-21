# ==========================================
# 1. BARRA LATERAL (NAVEGACIÓN)
# ==========================================
st.sidebar.title("🌀 Navegación")

# -- Categoría Principal --
categoria = st.sidebar.radio(
    "📂 Categoría:",
    ["Sistemas Dinámicos", "Fractales", "Cuencas de Atracción"]
)

st.sidebar.markdown("---")

# -- Sub-Menú (Depende de la categoría) --
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
        ("Fractal de Newton (Próximamente)",)
    )

# -- Sección de Referencias (Fija abajo) --
st.sidebar.markdown("---")
st.sidebar.write("### ℹ️ Info")
mostrar_referencias = st.sidebar.checkbox("Ver Bibliografía")


# ==========================================
# 2. LÓGICA DE VISUALIZACIÓN
# ==========================================

# CASO A: El usuario quiere ver las Referencias
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
    
    st.info("Desmarca la casilla 'Ver Bibliografía' en la barra lateral para volver a los gráficos.")

# CASO B: Visualización de Experimentos
else:
    # ---------------------------------------
    # SISTEMAS DINÁMICOS
    # ---------------------------------------
    if opcion == "Mapa Logístico (2D)":
        st.title("El Mapa Logístico")
        st.markdown(r"Ecuación: $x_{n+1} = r x_n (1 - x_n)$")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            n_iter = st.slider("Iteraciones", 500, 2000, 1000)
            r_range = st.slider("Rango de r", 2.5, 4.0, (2.5, 4.0))
        
        with col2:
            r = np.linspace(r_range[0], r_range[1], 1000)
            x = 1e-5 * np.ones(1000)
            for i in range(100): x = r * x * (1 - x) # Transitorio
            
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
        st.markdown(r"El sistema clásico de convección atmosférica.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            sigma = st.slider("Sigma", 0.0, 20.0, 10.0)
            rho = st.slider("Rho", 0.0, 50.0, 28.0)
            beta = st.slider("Beta", 0.0, 5.0, 2.66)
        
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
                line=dict(color=zs, colorscale='Viridis', width=2), opacity=0.8
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='#0E1117',
                scene=dict(bgcolor='#0E1117', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
            )
            st.plotly_chart(fig, use_container_width=True)

    elif opcion == "Atractor de Thomas (3D)":
        st.title("Atractor de Thomas")
        st.markdown("Atractor cíclicamente simétrico.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            b = st.slider("Beta (b)", 0.0, 1.0, 0.205)
            paleta = st.selectbox("Color", ("Ice", "Plasma", "Viridis", "Turbo"))
        
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
        st.markdown(r"Frontera del conjunto $z_{n+1} = z_n^2 + c$")
        
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

            with st.spinner('Calculando...'):
                plt.figure(figsize=(10, 10), facecolor='#0E1117')
                fractal = mandelbrot(resolucion, resolucion, iteraciones)
                plt.imshow(fractal, cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
                plt.axis('off')
                st.pyplot(plt)

    # ---------------------------------------
    # CUENCAS DE ATRACCIÓN (Placeholder)
    # ---------------------------------------
    elif opcion == "Fractal de Newton (Próximamente)":
        st.title("Fractal de Newton")
        st.info("🚧 Esta sección está en construcción.")
        st.markdown("""
        Aquí visualizaremos las **Cuencas de Atracción**: regiones del plano complejo que convergen a diferentes raíces de un polinomio.
        
        *Próximamente implementaremos el método de Newton-Raphson para $z^3 - 1 = 0$.*
        """)
