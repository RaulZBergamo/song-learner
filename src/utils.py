"""
Módulo com funções utilitárias para o projeto.
"""

# Dicionário de mapeamento das notas para números base
note_to_base_label = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

def convert_notes_to_labels(note: str) -> int:
    """
    Converte uma lista de notas musicais com oitavas em rótulos numéricos únicos.
    
    A fórmula para o rótulo é: (oitava * 12) + número da nota.
    
    :param notes: Lista de notas musicais com oitavas (ex: ['C4', 'D#5', 'E3'])
    :return: Lista de rótulos numéricos únicos correspondentes às notas com oitavas.
    """
    # Separar o nome da nota da oitava (ex: 'C4' -> 'C' e '4')
    note_name = note[:-1]  # Parte da nota, ex: 'C'
    octave = int(note[-1])  # Parte da oitava, ex: 4

    # Converter a nota para o rótulo base e ajustar pela oitava
    note_label = note_to_base_label[note_name]
    return (octave * 12) + note_label  # Calcula o rótulo único
