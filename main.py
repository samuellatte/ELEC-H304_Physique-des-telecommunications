import numpy as np
from scipy import constants
import time as t


### Paramètres à passer :
Emetteur = [np.array([19,1]),np.array([41,7])]  # Position des émetteurs
nbr_rebond = 1
resol_x = 1
resol_y = 1


# On commence par définir les murs et les couleurs :

murs = {'mur1': [np.array([12,8]),np.array([43,8])],
        'mur2': [np.array([43,8]),np.array([44,7])],
        'mur3': [np.array([44,7]),np.array([44,1])],
        'mur4': [np.array([44,1]),np.array([43,0])],
        'mur5': [np.array([43,0]),np.array([12,0])],

        # bas du nez
        'mur6': [np.array([12,0]),np.array([8,0.26217385126095305])],
        'mur7': [np.array([8,0.26217385126095305]), np.array([5,0.6571313440231403])],
        'mur8': [np.array([5,0.6571313440231403]),np.array([3,1.193029728885386])],
        'mur9': [np.array([3,1.193029728885386]), np.array([1.5,2.0590551326698243])],
        'mur10': [np.array([1.5,2.0590551326698243]), np.array([0.5,3.0590551326698243])],
        'mur11': [np.array([0.5,3.0590551326698243]), np.array([0,3.9250805364542627])],
        'mur12': [np.array([0,3.9250805364542627]), np.array([0,4.075080536454262])],

        #haut du nez
        'mur13': [np.array([12,8]), np.array([8,7.737826148739047])],
        'mur14': [np.array([8,7.737826148739047]), np.array([5,7.34286865597686])],
        'mur15': [np.array([5,7.34286865597686]), np.array([3,6.806970271114614])],
        'mur16': [np.array([3,6.806970271114614]), np.array([1.5,5.940944867330176])],
        'mur17': [np.array([1.5,5.940944867330176]), np.array([0.5,4.940944867330176])],
        'mur18': [np.array([0.5,4.940944867330176]), np.array([0,4.074919463545737])],
        'mur19': [np.array([0,4.074919463545737]), np.array([0,3.924919463545737])],

        #séparation entre cockpit-sièges, entre classes et parois du fond de l'avion
        'mur20': [np.array([8,7.737826148739047]), np.array([8,0.26217385126095305])],
        'mur21': [np.array([12,8]), np.array([12,5])],
        'mur22': [np.array([12,0]), np.array([12,3])],
        'mur23': [np.array([20,8]), np.array([20,5])],
        'mur24': [np.array([20,0]), np.array([20,3])],
        'mur25': [np.array([38,8]), np.array([38,5])],
        'mur26': [np.array([38,0]), np.array([38,3])],
        'mur27': [np.array([40,8]), np.array([40,5])],
        'mur28': [np.array([40,0]), np.array([40,3])],

        #sièges de droite
        'mur29': [np.array([14,8]), np.array([14,5])],
        'mur30': [np.array([16,8]), np.array([16,5])],
        'mur31': [np.array([18,8]), np.array([18,5])],
        'mur32': [np.array([22,8]), np.array([22,5])],
        'mur33': [np.array([23.5,8]), np.array([23.5,5])],
        'mur34': [np.array([25,8]), np.array([25,5])],
        'mur35': [np.array([26.5,8]), np.array([26.5,5])],
        'mur36': [np.array([28,8]), np.array([28,5])],
        'mur37': [np.array([29.5,8]), np.array([29.5,5])],
        'mur38': [np.array([31,8]), np.array([31,5])],
        'mur39': [np.array([32.5,8]), np.array([32.5,5])],
        'mur40': [np.array([34,8]), np.array([34,5])],
        'mur41': [np.array([35.5,8]), np.array([35.5,5])],
        'mur42': [np.array([37,8]), np.array([37,5])],

        #sièges de gauche
        'mur43': [np.array([14,0]), np.array([14,3])],
        'mur44': [np.array([16,0]), np.array([16,3])],
        'mur45': [np.array([18,0]), np.array([18,3])],
        'mur46': [np.array([22,0]), np.array([22,3])],
        'mur47': [np.array([23.5,0]), np.array([23.5,3])],
        'mur48': [np.array([25,0]), np.array([25,3])],
        'mur49': [np.array([26.5,0]), np.array([26.5,3])],
        'mur50': [np.array([28,0]), np.array([28,3])],
        'mur51': [np.array([29.5,0]), np.array([29.5,3])],
        'mur52': [np.array([31,0]), np.array([31,3])],
        'mur53': [np.array([32.5,0]), np.array([32.5,3])],
        'mur54': [np.array([34,0]), np.array([34,3])],
        'mur55': [np.array([35.5,0]), np.array([35.5,3])],
        'mur56': [np.array([37,0]), np.array([37,3])]}

# On définit ensuite les variables
resist = 73
f = 60*1e9
w = 2*constants.pi*f
l = 0.10
mu_0 = constants.mu_0
epsilon_0 = constants.epsilon_0
Z_0 = np.sqrt(mu_0/epsilon_0)
c = constants.c
beta = w/c
lambd = (constants.c)/f
Ptx_dbm = 20
Ptx = 10**(Ptx_dbm/10)
Gtx = 16/(3*np.pi)
PtxGtx = Ptx*Gtx*(10**-3) # En Watt
longueur_avion = 44
largeur_avion = 8
Puissances_dBm = []
Debit_binaire = []


# Paramètres des sièges
sigma_plastique = 0.003
epsilon_r_plastique = 2.25
epsilon_plastique = epsilon_r_plastique*epsilon_0
Z_m_plastique = np.sqrt(mu_0/(epsilon_plastique-(sigma_plastique/w)*1j))
alpha_m_plastique = w*np.sqrt((mu_0*epsilon_plastique)/2)*(np.sqrt(np.sqrt(1+(sigma_plastique/(w*epsilon_plastique))**2)-1))
beta_m_plastique = w*np.sqrt((mu_0*epsilon_plastique)/2)*(np.sqrt(np.sqrt(1+(sigma_plastique/(w*epsilon_plastique))**2)+1))
gamma_m_plastique = alpha_m_plastique + beta_m_plastique*1j


# Paramètres des parois
sigma_GRP = 0.868
epsilon_r_GRP = 8.7
epsilon_GRP = epsilon_r_GRP*epsilon_0
Z_m_GRP = np.sqrt(mu_0/(epsilon_GRP-(sigma_GRP/w)*1j))
alpha_m_GRP = w*np.sqrt((mu_0*epsilon_GRP)/2)*(np.sqrt(np.sqrt(1+(sigma_GRP/(w*epsilon_GRP))**2)-1))
beta_m_GRP = w*np.sqrt((mu_0*epsilon_GRP)/2)*(np.sqrt(np.sqrt(1+(sigma_GRP/(w*epsilon_GRP))**2)+1))
gamma_m_GRP = alpha_m_GRP + beta_m_GRP*1j


def calc_param():
    for indice,cle in enumerate(murs):
        u = np.array(murs[cle][1]-np.array(murs[cle][0]))
        u = u / np.linalg.norm(u)
        n = np.array([u[1], -u[0]])  # Permet de calculer le vecteur normal de chaque mur
        murs[cle].append(u)  # Position 2 dans la liste
        murs[cle].append(n)  # Position 3 dans la liste
        if indice <= 27: # On ajoute les paramètres du mur cloison en GRP
            murs[cle].append(epsilon_r_GRP) # Position 4 dans la liste
            murs[cle].append(Z_m_GRP)  # Position 5 dans la liste
            murs[cle].append(gamma_m_GRP)  # Position 6 dans la liste
        else:
            murs[cle].append(epsilon_r_plastique)  # Position 4 dans la liste
            murs[cle].append(Z_m_plastique)  # Position 5 dans la liste
            murs[cle].append(gamma_m_plastique)  # Position 6 dans la liste


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
    sin_theta_t = np.sqrt(1 / murs[cle][4]) * sin_theta_i
    cos_theta_t = np.sqrt(1 - sin_theta_t ** 2)
    s = l / cos_theta_t
    gamma_pol_perp = (murs[cle][5] * cos_theta_i - Z_0 * cos_theta_t) / (murs[cle][5] * cos_theta_i + Z_0 * cos_theta_t)
    T_m = T_m * ((1 - gamma_pol_perp ** 2) * np.exp(-murs[cle][6] * s)) / (1 - (gamma_pol_perp ** 2) * np.exp(-2 * murs[cle][6] * s) * np.exp(beta * 2 * s * sin_theta_t * sin_theta_i * 1j))
    return T_m


def calc_coeff_reflexion(d,cle):
    norm_d = np.linalg.norm(d)
    d_unit = d/norm_d
    cos_theta_i = np.linalg.norm(np.dot(d_unit, murs[cle][3]))
    sin_theta_i = np.sqrt(1 - cos_theta_i ** 2)
    sin_theta_t = np.sqrt(1 / murs[cle][4]) * sin_theta_i
    cos_theta_t = np.sqrt(1 - sin_theta_t ** 2)
    s = l / cos_theta_t
    gamma_pol_perp = (murs[cle][5] * cos_theta_i - Z_0 * cos_theta_t) / (murs[cle][5] * cos_theta_i + Z_0 * cos_theta_t)
    coeff_refl = gamma_pol_perp - (1 - gamma_pol_perp ** 2) * (((gamma_pol_perp) * np.exp(-2 * murs[cle][6] * s) * np.exp(beta * 2 * s * sin_theta_t * sin_theta_i * 1j)) / (1 - (gamma_pol_perp ** 2) * np.exp(-2 * murs[cle][6] * s) * np.exp(beta * 2 * s * sin_theta_t * sin_theta_i * 1j)))
    return coeff_refl


def trajet_direct(TX,RX):
    if np.array_equal(TX,np.array([19,1.5])):
        T_m = verif_transmission(TX,RX)
        d = [RX[0] - TX[0], RX[1] - TX[1]]  # On définit le vecteur d
        norm_d = np.linalg.norm(d)
        E0 = T_m * np.sqrt(60 * PtxGtx) * (np.exp(-beta * norm_d * 1j) / (norm_d))
        P_RX_Watt = ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E0) ** 2  # Puissance reçue grâce à l'onde direct, en [W]
        P_RX_dBm = 10 * np.log10(P_RX_Watt * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
        # print('E0 = ',E0)
        # print('P_RX_Watt = ', P_RX_Watt)
        # print('P_RX_dBm = ', P_RX_dBm)
        # print('--------')
        # plt.plot([TX[0], RX[0]], [TX[1], RX[1]], color="b")
        # plt.plot(TX[0], TX[1], marker="*", color='g', label='Emetteur 1')
        # plt.legend(loc="lower right")
    else:
        T_m = verif_transmission(TX, RX)
        d = [RX[0] - TX[0], RX[1] - TX[1]]  # On définit le vecteur d
        norm_d = np.linalg.norm(d)
        E0 = T_m * np.sqrt(60 * PtxGtx) * (np.exp(-beta * norm_d * 1j) / (norm_d))
        P_RX_Watt = ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E0) ** 2  # Puissance reçue grâce à l'onde direct, en [W]
        P_RX_dBm = 10 * np.log10(P_RX_Watt * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
        # print('E0 = ',E0)
        # print('P_RX_Watt0 = ', P_RX_Watt)
        # print('P_RX_dBm = ', P_RX_dBm)
        # print('--------')
        # plt.plot([TX[0], RX[0]], [TX[1], RX[1]], color="b")
        # plt.plot(TX[0], TX[1], marker="*", color='g', label='Emetteur 2')
        # plt.plot(RX[0], RX[1], marker="*", color='y', label='Récepteur')
        # plt.legend(loc="lower right")
    return P_RX_Watt


def calc_image(TX,RX,nbr_rebond):
    P_RX_Watt_1 = 0
    P_RX_Watt_2 = 0
    P_RX_Watt_3 = 0
    if nbr_rebond >= 0 : # On calcule seulement le trajet direct
        P_RX_Watt_0 = trajet_direct(TX,RX)
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
                        P_RX_Watt_1 += ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E1) ** 2  # Puissance reçue grâce à l'onde direct, en [mW]
                        P_RX_dBm = 10 * np.log10(P_RX_Watt_1 * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
                        # print('E1 = ', E1)
                        # print('P_RX_Watt1 = ', P_RX_Watt)
                        # print('P_RX_dBm = ', P_RX_dBm)
                        # print('--------')
                        # plt.plot([TX[0], P1[0]], [TX[1], P1[1]], 'b', lw=0.5)
                        # plt.plot([P1[0], RX[0]], [P1[1], RX[1]], 'b', lw=0.5)
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
                                            if murs[cle][2][0] * d1[1] != murs[cle][2][1] * d1[0]:
                                                t1 = (d1[1] * (image1[0] - murs[cle][0][0]) - d1[0] * (image1[1] - murs[cle][0][1])) / (murs[cle][2][0] * d1[1] - murs[cle][2][1] * d1[0])
                                                if 0 < t1 < np.linalg.norm(murs[cle][1] - murs[cle][0]):
                                                    P1 = murs[cle][0] + t1 * murs[cle][2]
                                                    T_m_TX_P1 = verif_transmission(TX, P1)
                                                    T_m_P1_P2 = verif_transmission(P1, P2)
                                                    T_m_P2_RX = verif_transmission(P2, RX)
                                                    coeff_reflexion_1 = calc_coeff_reflexion(d1, cle)
                                                    coeff_reflexion_2 = calc_coeff_reflexion(d2, cle2)
                                                    E2 = coeff_reflexion_1 * coeff_reflexion_2 * T_m_TX_P1 * T_m_P1_P2 * T_m_P2_RX * np.sqrt(60 * PtxGtx) * (np.exp(-beta * norm_d2 * 1j) / (norm_d2))
                                                    P_RX_Watt_2 += ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E2) ** 2  # Puissance reçue grâce à l'onde direct, en [mW]
                                                    P_RX_dBm = 10 * np.log10(P_RX_Watt_2 * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
                                                    # print('d = ',norm_d2)
                                                    # print("coeff transmission =", T_m_TX_P1 * T_m_P1_P2 * T_m_P2_RX)
                                                    # print("coeff réflexion1 =", coeff_reflexion_1)
                                                    # print("coeff réflexion2 =", coeff_reflexion_2)
                                                    # print("E2 =", E2)
                                                    # print('P_RX_Watt2 = ', P_RX_Watt)
                                                    # print('P_RX_dBm = ', P_RX_dBm)
                                                    # print('--------')
                                                    # plt.plot([TX[0], P1[0]], [TX[1], P1[1]], 'y', lw=0.5)
                                                    # plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'y', lw=0.5)
                                                    # plt.plot([P2[0], RX[0]], [P2[1], RX[1]], 'y', lw=0.5)
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
                                                                        if murs[cle][2][0] * d1[1] != murs[cle][2][1] * d1[0]:
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
                                                                                P_RX_Watt_3 += ((lambd ** 2) / (8 * (constants.pi ** 2) * resist)) * np.linalg.norm(E3) ** 2  # Puissance reçue grâce à l'onde direct, en [mW]
                                                                                P_RX_dBm = 10 * np.log10(P_RX_Watt_3 * 1e3)  # Puissance reçue grâce à l'onde direct, en dBm
                                                                                # print('d = ', norm_d3)
                                                                                # print("coeff transmission =",T_m_TX_P1 * T_m_P1_P2 * T_m_P2_P3 * T_m_P3_RX)
                                                                                # print("coeff réflexion1 =", coeff_reflexion_1)
                                                                                # print("coeff réflexion2 =", coeff_reflexion_2)
                                                                                # print("coeff réflexion3 =", coeff_reflexion_3)
                                                                                # print("E3 =", E3)
                                                                                # print('P_RX_Watt3 = ', P_RX_Watt)
                                                                                # print('P_RX_dBm = ', P_RX_dBm)
                                                                                # print('--------')
                                                                                # plt.plot([TX[0], P1[0]], [TX[1], P1[1]], 'g', lw=0.5)
                                                                                # plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'g', lw=0.5)
                                                                                # plt.plot([P2[0], P3[0]], [P2[1], P3[1]], 'g', lw=0.5)
                                                                                # plt.plot([P3[0], RX[0]], [P3[1], RX[1]], 'g', lw=0.5)
    P_RX_Watt = P_RX_Watt_0 + P_RX_Watt_1 + P_RX_Watt_2 + P_RX_Watt_3
    # print('P_RX_Watt', P_RX_Watt)
    return P_RX_Watt


def main():
    t0 = t.time()
    # Partie calcul des paramètres :
    calc_param()

    # Partie calcul des rebonds :
    # On itère sur les récepteurs au centre des rectangles de côté Resol_x/Resol_y
    for i in np.arange(largeur_avion-resol_y/2, -resol_y/2, -resol_y): # On itère sur les lignes, de resol_y/2 à largeur_avion-(resol_y)/2
        for j in np.arange(resol_x/2, longueur_avion+(resol_x/2), resol_x): # On itère sur les colonnes
            # On se déplace de ligne en ligne (dernière ligne de 0.5 à 43.5, puis avant dernière ligne de 0.5 à 43.5 etc)
            RX = [j, i]
            # print('RX = ',RX)
            P_RX_Watt = 0
            for TX in Emetteur: # On itère sur les émetteurs
                P_RX_Watt += calc_image(TX, RX, nbr_rebond)
            P_RX_dBm = 10 * np.log10(P_RX_Watt * 1e3)
            Deb_bin = 183.7*P_RX_dBm + 14356.1
            Puissances_dBm.append(P_RX_dBm)
            Debit_binaire.append(Deb_bin)

    # Partie sauvegarde des données :
    # if nbr_rebond == 0:
    #     np.savetxt('P_RX_dBm-0_3TX', Puissances_dBm)
    #     np.savetxt('Debit_bin-0_3TX', Debit_binaire)
    # if nbr_rebond == 1:
    #     np.savetxt('P_RX_dBm-1_3TX', Puissances_dBm)
    #     np.savetxt('Debit_bin-1_3TX', Debit_binaire)
    # if nbr_rebond == 2:
    #     np.savetxt('P_RX_dBm-2_3TX', Puissances_dBm)
    #     np.savetxt('Debit_bin-2_3TX', Debit_binaire)
    # if nbr_rebond == 3:
    #     np.savetxt('P_RX_dBm-3_3TX', Puissances_dBm)
    #     np.savetxt('Debit_bin-3_3TX', Debit_binaire)

    t1 = t.time()
    print('Temps exécution =',(t1 - t0),'secondes')
main()