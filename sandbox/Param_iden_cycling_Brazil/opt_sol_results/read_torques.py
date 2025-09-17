import json

def ler_torques(path_arquivo):
    with open(path_arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados

def imprimir_torques(dados):
    for musculo, ang_dict in dados.items():
        print(f"== Músculo: {musculo} ==")
        for angulo, valores in ang_dict.items():
            print(f"  Ângulo: {angulo}°")
            print(f"    Torque medido  ({len(valores['torque_medido'])} amostras):")
            print(f"      {valores['torque_medido'][:5]} ...")  # mostra só primeiras 5
            print(f"    Torque predito ({len(valores['torque_predito'])} amostras):")
            print(f"      {valores['torque_predito'][:5]} ...")
        print()

# exemplo de uso:
arquivo = "pickles/torques_crankset.json"
dados = ler_torques(arquivo)
imprimir_torques(dados)