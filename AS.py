import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# NIE WIEM CZY PARAMETRY R SĄ ODPOWANIE
def bray_liebhafsky_model(t, y, R1, R2, R3, R4, R5, R6, R7):
    """Model równań różniczkowych reakcji cos tam jodu
    :param t: czas ustawiany przez solver
    :param y: wartości początkowe a później wektory stanu
     U0: wartość początkowa Kwasu jodowego III HIO2
     V0: wart. począt. Anionu jodkowego
     Z0: wart. począt. Jodu cząsteczkowego
     W0: wart. począt. tlenu cząsteczkowego 02
    :param R1: szybkość tworzenia HIO₂ z IO₃⁻ i I⁻
    :param R2: to szybkość zużycia HIO₂ przez reakcję z I⁻
    :param R3: tempo autokatalitycznego tworzenia HIO₂
    :param R4: tempo rozpadu HIO₂
    :param R5: tempo powstawania I⁻ z I₂
    :param R6: tempo usuwania tlenu (O₂)
    :param R7: bla bla bla
    """
    U, V, Z, W = y  # rozpakowujemy listę

    dUdt = R1 * V + R3 * U - R2 * U * V - R4 * U ** 2
    dVdt = R5 * Z - R1 * V - R2 * U * V
    dZdt = R3 * U - (R5 + R7) * Z
    dWdt = -R6 * W
    return [dUdt, dVdt, dZdt, dWdt]


def bl_model(
        U0, V0, Z0, W0,
        R1, R2, R3, R4, R5, R6, R7,
        t_max=3000,
        n_points=5000,
        method='RK45',
        plot=True
):
    """
    Symulacja modelu Bray–Liebhafsky'ego (4 zmienne)

    Parametry stanu:
    U0 – HIO2
    V0 – I-
    Z0 – I2
    W0 – O2

    Parametry kinetyczne:
    R1–R7 – parametry modelu

    Parametry numeryczne:
    t_max – czas końcowy
    n_points – liczba punktów czasowych
    method – metoda całkowania
    plot – czy rysować wykres
    """

    y0 = [U0, V0, Z0, W0]
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_points)

    sol = solve_ivp(
        bray_liebhafsky_model,
        t_span,
        y0,
        args=(R1, R2, R3, R4, R5, R6, R7),
        t_eval=t_eval,
        method=method
    )

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))

        axs[0, 0].plot(sol.t, sol.y[0])
        axs[0, 0].set_title('U – HIO₂')

        axs[0, 1].plot(sol.t, sol.y[1])
        axs[0, 1].set_title('V – I⁻')

        axs[1, 0].plot(sol.t, sol.y[2])
        axs[1, 0].set_title('Z – I₂')

        axs[1, 1].plot(sol.t, sol.y[3])
        axs[1, 1].set_title('W – O₂')

        for ax in axs.flat:
            ax.set_xlabel('Czas')
            ax.set_ylabel('Stężenie')
            ax.grid(True)

        fig.suptitle('Model Bray–Liebhafsky (4 zmienne)', fontsize=14)
        fig.tight_layout()
        plt.show()


# PARAMETRY

U0 = 1000
V0 = 995
Z0 = 3
W0 = 2

R1 = 0.0035
R2 = 1.0
R3 = 1.99
R4 = 0.0028
R5 = 1.0
R6 = 0.0017
R7 = 0.02
bl_model(U0, V0, Z0, W0, R1, R2, R3, R4, R5, R6, R7)
