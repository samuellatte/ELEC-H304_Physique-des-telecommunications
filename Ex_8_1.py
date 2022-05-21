import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

# On commence par définir les murs :

murs = {'mur1': [np.array([0,0]), np.array([0,80])], # On trace le mur vertical
        'mur2': [np.array([0,80]), np.array([70,80])], # On trace le mur horizontal haut
        'mur3': [np.array([0,20]), np.array([70,20])]} # On trace le mur horizontal moyen
        # 'mur4': [np.array([0,40]), np.array([70,40])], # On trace le mur horizontal bas
        # 'mur5': [np.array([0,60]), np.array([70,60])]}

# On définit ensuite les variables

resist = 73
f = 868.3*1e6
w = 2*constants.pi*f
sigma = 0.018
l = 0.15
mu_0 = constants.mu_0
c = constants.c
epsilon_0 = constants.epsilon_0
epsilon_r = 4.8
epsilon = epsilon_r*epsilon_0
Z_m = np.sqrt(mu_0/(epsilon-(sigma/w)*1j))
Z_0 = np.sqrt(mu_0/epsilon_0)
alpha_m = w*np.sqrt((mu_0*epsilon)/2)*(np.sqrt(np.sqrt(1+(sigma/(w*epsilon))**2)-1))
beta_m = w*np.sqrt((mu_0*epsilon)/2)*(np.sqrt(np.sqrt(1+(sigma/(w*epsilon))**2)+1))
beta = w/c
gamma_m = alpha_m + beta_m*1j
PtxGtx_dBm = 2.15
PtxGtx = (10 ** (PtxGtx_dBm / 10))*1e-3
lambd = (constants.c)/f


def calc_param():
    for cle in murs:
        u = np.array(murs[cle][1]-np.array(murs[cle][0]))
        u = u / np.linalg.norm(u)
        n = np.array([u[1], -u[0]])  # Permet de calculer le vecteur normal de chaque mur
        murs[cle].append(u)  # Position 2 dans la liste
        murs[cle].append(n)  # Position 3 dans la liste


def verif_transmission(TX,RX):
    T_m = 1
    d = [RX[0] - TX[0], RX[1] - TX[1]]  # On définit le vecteur d
    for cle in murs:  # On vérifie quel mur est traversé par le trajet direct
        if murs[cle][2][0] * d[1] != murs[cle][2][1] * d[0]:
            t = (d[1] * (TX[0] - murs[cle][0][0]) - d[0] * (TX[1] - murs[cle][0][1])) / (murs[cle][2][0] * d[1] - murs[cle][2][1] * d[0])
            s = TX - murs[cle][0]
            if 0 < t < np.linalg.norm(murs[cle][1] - murs[cle][0]):  # Vérifie s'il y a une transmission
                if np.sign(np.dot(s, murs[cle][3])) != np.sign(np.dot(RX - murs[cle][0], murs[cle][3])) and np.sign(np.dot(RX - murs[cle][0], murs[cle][3])) != 0 and np.sign(np.dot(s, murs[cle][3])) != 0:
                    P = murs[cle][0] + t * murs[cle][2]
                    # print('Pt = ', P)
                    T_m = calc_coeff_transmission(d,cle,T_m)
    return T_m


def calc_coeff_transmission(d,cle,T_m):
    norm_d = np.linalg.norm(d)
    d_unit = d/norm_d
    cos_theta_i = np.linalg.norm(np.dot(d_unit, murs[cle][3]))
    sin_theta_i = np.sqrt(1 - cos_theta_i ** 2)
    sin_theta_t = np.sqrt(1 / epsilon_r) * sin_theta_i
    cos_theta_t = np.sqrt(1 - sin_theta_t ** 2)
    s = l / cos_theta_t
    gamma_pol_perp = (Z_m * cos_theta_i - Z_0 * cos_theta_t) / (Z_m * cos_theta_i + Z_0 * cos_theta_t)
    T_m = T_m * ((1 - gamma_pol_perp ** 2) * np.exp(-gamma_m * s)) / (1 - (gamma_pol_perp ** 2) * np.exp(-2 * gamma_m * s) * np.exp(beta * 2 * s * sin_theta_t * sin_theta_i * 1j))
    return T_m


def calc_coeff_reflexion(d,cle):
    norm_d = np.linalg.norm(d)
    d_unit = d/norm_d
    cos_theta_i = np.linalg.norm(np.dot(d_unit, murs[cle][3]))
    sin_theta_i = np.sqrt(1 - cos_theta_i ** 2)
    sin_theta_t = np.sqrt(1 / epsilon_r) * sin_theta_i
    cos_theta_t = np.sqrt(1 - sin_theta_t ** 2)
    s = l / cos_theta_t
    gamma_pol_perp = (Z_m * cos_theta_i - Z_0 * cos_theta_t) / (Z_m * cos_theta_i + Z_0 * cos_theta_t)
    coeff_refl = gamma_pol_perp - (1 - gamma_pol_perp ** 2) * (((gamma_pol_perp) * np.exp(-2 * gamma_m * s) * np.exp(beta * 2 * s * sin_theta_t * sin_theta_i * 1j)) / (1 - (gamma_pol_perp ** 2) * np.exp(-2 * gamma_m * s) * np.exp(beta * 2 * s * sin_theta_t * sin_theta_i * 1j)))
    return coeff_refl


def trajet_direct(TX,RX):
    T_m = verif_transmission(TX,RX)
    d = [RX[0] - TX[0], RX[1] - TX[1]]  # On définit le vecteur d
    norm_d = np.linalg.norm(d)
    E0 = T_m * np.sqrt(60 * PtxGtx) * (np.exp(-beta * norm_d * 1j) / (norm_d))
    P_RX_Watt = ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E0) ** 2  # Puissance reçue grâce à l'onde direct, en [W]
    P_RX_dBm = 10 * np.log10(P_RX_Watt * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
    print('Coeff transmission = ', T_m)
    print('E0 = ',E0, 'V/m')
    print('P_RX_Watt = ', P_RX_Watt, 'W')
    print('P_RX_dBm = ', P_RX_dBm, 'dBm')
    print('--------')
    plt.plot([TX[0], RX[0]], [TX[1], RX[1]], color="b")
    plt.plot(TX[0], TX[1], marker="*", color='r', label='Emetteur')
    plt.plot(RX[0], RX[1], marker="*", color='b', label='Récepteur')
    plt.legend(loc="best")
    return E0


def calc_image(TX,RX,nbr_rebond):
    if nbr_rebond >= 0 : # On calcule seulement le trajet direct
        E0 = trajet_direct(TX,RX)
        if 1 <= nbr_rebond : # On calcule pour 1 rebond
            for cle in murs:
                s1 = TX - murs[cle][0]
                image1 = TX - 2 * np.dot(s1, murs[cle][3]) * murs[cle][3]
                d1 = RX - image1
                norm_d1 = np.linalg.norm(d1)
                if murs[cle][2][0] * d1[1] != murs[cle][2][1] * d1[0]: # Condition pour éviter que t1 soit infini
                    t1 = (d1[1] * (image1[0] - murs[cle][0][0]) - d1[0] * (image1[1] - murs[cle][0][1])) / (murs[cle][2][0] * d1[1] - murs[cle][2][1] * d1[0])
                    if 0 < t1 < np.linalg.norm(murs[cle][1] - murs[cle][0]) and np.sign(np.dot(s1, murs[cle][3])) == np.sign(np.dot(RX - murs[cle][0], murs[cle][3])): # Si t1 est dans le mur et que la condition avec s et n est vérifiée
                        P1 = murs[cle][0] + t1 * murs[cle][2] # On a donc un point de rélflexion
                        T_m_TX_P = verif_transmission(TX, P1)
                        T_m_P_RX = verif_transmission(P1, RX)
                        coeff_reflexion_1 = calc_coeff_reflexion(d1,cle)
                        E1 = coeff_reflexion_1 * T_m_TX_P * T_m_P_RX * np.sqrt(60 * PtxGtx) * (np.exp(-beta * norm_d1 * 1j) / (norm_d1))
                        P_RX_Watt = ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E0+E1) ** 2  # Puissance reçue grâce à l'onde direct, en [W]
                        P_RX_dBm = 10 * np.log10(P_RX_Watt * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
                        print("coeff transmission =", T_m_TX_P * T_m_P_RX)
                        print("coeff réflexion =", coeff_reflexion_1)
                        print('E1 = ', E1, 'V/m')
                        print('P_RX_Watt = ', P_RX_Watt, 'W')
                        print('P_RX_dBm = ', P_RX_dBm, 'dBm')
                        print('--------')
                        plt.plot([TX[0], P1[0]], [TX[1], P1[1]], 'b', lw=0.5)
                        plt.plot([P1[0], RX[0]], [P1[1], RX[1]], 'b', lw=0.5)
                        if 2 <= nbr_rebond:
                            for cle2 in murs:  # On itère sur chaque mur
                                if cle2 != cle:  # Condition pour ne pas retomber sur une image2 qui soit TX
                                    s2 = image1 - murs[cle2][0]
                                    image2 = image1 - 2 * np.dot(s2, murs[cle2][3]) * murs[cle2][3]
                                    d2 = RX - image2
                                    norm_d2 = np.linalg.norm(d2)
                                    if murs[cle2][2][0] * d2[1] != murs[cle2][2][1] * d2[0]: # Condition pour éviter que t2 soit infini
                                        t2 = (d2[1] * (image2[0] - murs[cle2][0][0]) - d2[0] * (image2[1] - murs[cle2][0][1])) / (murs[cle2][2][0] * d2[1] - murs[cle2][2][1] * d2[0])
                                        if 0 < t2 < np.linalg.norm(murs[cle2][1] - murs[cle2][0]) and np.sign(np.dot(s2, murs[cle2][3])) == np.sign(np.dot(RX - murs[cle2][0], murs[cle2][3])):  # Si t2 est dans le mur et que la condition avec s et n est vérifiée
                                            P2 = murs[cle2][0] + t2 * murs[cle2][2]  # On a donc un point de rélflexion
                                            d1 = P2 - image1
                                            t1 = (d1[1] * (image1[0] - murs[cle][0][0]) - d1[0] * (image1[1] - murs[cle][0][1])) / (murs[cle][2][0] * d1[1] - murs[cle][2][1] * d1[0])
                                            if 0 < t1 < np.linalg.norm(murs[cle][1] - murs[cle][0]):
                                                P1 = murs[cle][0] + t1 * murs[cle][2]
                                                T_m_TX_P1 = verif_transmission(TX, P1)
                                                T_m_P1_P2 = verif_transmission(P1, P2)
                                                T_m_P2_RX = verif_transmission(P2, RX)
                                                coeff_reflexion_1 = calc_coeff_reflexion(d1, cle)
                                                coeff_reflexion_2 = calc_coeff_reflexion(d2, cle2)
                                                E2 = coeff_reflexion_1 * coeff_reflexion_2 * T_m_TX_P1 * T_m_P1_P2 * T_m_P2_RX * np.sqrt(60 * PtxGtx) * (np.exp(-beta * norm_d2 * 1j) / (norm_d2))
                                                P_RX_Watt = ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E0 + E2) ** 2  # Puissance reçue grâce à l'onde direct, en [W]
                                                P_RX_dBm = 10 * np.log10(P_RX_Watt * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
                                                # print('d = ',norm_d2)
                                                print("coeff transmission =", T_m_TX_P1 * T_m_P1_P2 * T_m_P2_RX)
                                                print("coeff réflexion1 =", coeff_reflexion_1)
                                                print("coeff réflexion2 =", coeff_reflexion_2)
                                                print("E2 =", E2, 'V/m')
                                                print('P_RX_Watt = ', P_RX_Watt, 'W')
                                                print('P_RX_dBm = ', P_RX_dBm, 'dBm')
                                                print('--------')
                                                plt.plot([TX[0], P1[0]], [TX[1], P1[1]], 'y', lw=0.5)
                                                plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'y', lw=0.5)
                                                plt.plot([P2[0], RX[0]], [P2[1], RX[1]], 'y', lw=0.5)
                                        if 3 <= nbr_rebond < 4:
                                            for cle3 in murs:  # On itère sur chaque mur
                                                if cle3 != cle2:  # Condition pour ne pas retomber sur une image3 qui soit image2
                                                    s3 = image2 - murs[cle3][0]
                                                    image3 = image2 - 2 * np.dot(s3, murs[cle3][3]) * murs[cle3][3]
                                                    d3 = RX - image3
                                                    norm_d3 = np.linalg.norm(d3)
                                                    if murs[cle3][2][0] * d3[1] != murs[cle3][2][1] * d3[0]:
                                                        t3 = (d3[1] * (image3[0] - murs[cle3][0][0]) - d3[0] * (image3[1] - murs[cle3][0][1])) / (murs[cle3][2][0] * d3[1] - murs[cle3][2][1] * d3[0]) # Condition pour éviter que t3 soit infini
                                                        if 0 < t3 < np.linalg.norm(murs[cle3][1] - murs[cle3][0]) and np.sign(np.dot(s3, murs[cle3][3])) == np.sign(np.dot(RX - murs[cle3][0], murs[cle3][3])):
                                                            P3 = murs[cle3][0] + t3 * murs[cle3][2]
                                                            d2 = P3 - image2
                                                            if murs[cle2][2][0] * d2[1] != murs[cle2][2][1] * d2[0]:
                                                                t2 = (d2[1] * (image2[0] - murs[cle2][0][0]) - d2[0] * (image2[1] - murs[cle2][0][1])) / (murs[cle2][2][0] * d2[1] - murs[cle2][2][1] * d2[0])
                                                                if 0 < t2 < np.linalg.norm(murs[cle2][1] - murs[cle2][0]):
                                                                    P2 = murs[cle2][0] + t2 * murs[cle2][2]
                                                                    d1 = P2 - image1
                                                                    t1 = (d1[1] * (image1[0] - murs[cle][0][0]) - d1[0] * (image1[1] - murs[cle][0][1])) / (murs[cle][2][0] * d1[1] - murs[cle][2][1] * d1[0])
                                                                    if 0 < t1 < np.linalg.norm(murs[cle][1] - murs[cle][0]):
                                                                        P1 = murs[cle][0] + t1 * murs[cle][2]
                                                                        T_m_TX_P1 = verif_transmission(TX, P1)
                                                                        T_m_P1_P2 = verif_transmission(P1, P2)
                                                                        T_m_P2_P3 = verif_transmission(P2, P3)
                                                                        T_m_P3_RX = verif_transmission(P3, RX)
                                                                        coeff_reflexion_1 = calc_coeff_reflexion(d1, cle)
                                                                        coeff_reflexion_2 = calc_coeff_reflexion(d2, cle2)
                                                                        coeff_reflexion_3 = calc_coeff_reflexion(d3, cle3)
                                                                        E3 = coeff_reflexion_1 * coeff_reflexion_2 * coeff_reflexion_3 * T_m_TX_P1 * T_m_P1_P2 * T_m_P2_P3 * T_m_P3_RX * np.sqrt(60 * PtxGtx) * (np.exp(-beta * norm_d3 * 1j) / (norm_d3))
                                                                        P_RX_Watt = ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E0 + E3) ** 2  # Puissance reçue grâce à l'onde direct, en [W]
                                                                        P_RX_dBm = 10 * np.log10(P_RX_Watt * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
                                                                        # print('d = ', norm_d3)
                                                                        print("coeff transmission =",T_m_TX_P1 * T_m_P1_P2 * T_m_P2_P3 * T_m_P3_RX)
                                                                        print("coeff réflexion1 =", coeff_reflexion_1)
                                                                        print("coeff réflexion2 =", coeff_reflexion_2)
                                                                        print("coeff réflexion3 =", coeff_reflexion_3)
                                                                        print("E3 =", E3, 'V/m')
                                                                        print('P_RX_Watt = ', P_RX_Watt, 'W')
                                                                        print('P_RX_dBm = ', P_RX_dBm, 'dBm')
                                                                        print('--------')
                                                                        plt.plot([TX[0], P1[0]], [TX[1], P1[1]], 'g', lw=0.5)
                                                                        plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'g', lw=0.5)
                                                                        plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'g', lw=0.5)
                                                                        plt.plot([P3[0], RX[0]], [P3[1], RX[1]], 'g', lw=0.5)


def dessine_murs():
    for cle in murs:
        plt.plot([murs[cle][0][0],murs[cle][1][0]],[murs[cle][0][1],murs[cle][1][1]],'r')


def main():
    # Partie définition des paramètres :
    # TX,RX,nbr_rebond = pos_TX_RX_rebond()
    TX = np.array([32,10])  # Position de l'émetteur
    RX = np.array([47,65])  # Position du récepteur
    nbr_rebond = 2
    calc_param()

    # Partie calcul des rebonds :
    calc_image(TX,RX,nbr_rebond)

    # Partie dessin des rebonds et murs :
    dessine_murs()
    # plt.axis('equal')
    plt.grid()
    plt.show()

main()