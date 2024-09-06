"""
Módulo para fazer a conversão dos número de pitch MIDI em nomes de notas musicais.
"""

class MidiConverter:
    """
    Classe responsável por converter números MIDI em nomes de notas musicais.
    """

    def __init__(self):
        # Definir as notas padrão
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def midi_to_note_name(self, midi_number):
        """
        Converte um número MIDI em uma nota musical no formato padrão.
        :param midi_number: Número MIDI (int)
        :return: Nome da nota (str)
        """
        octave = (midi_number // 12) - 1
        note = self.note_names[midi_number % 12]
        return f"{note}{octave}"

    def process_note_data(self, note_data):
        """
        Processa dados de uma nota extraída de um JSON e obtém o nome da nota MIDI.
        :param note_data: Dicionário com informações da nota (ex. pitch)
        :return: Nome da nota correspondente ao pitch
        """
        midi_number = note_data.get('pitch', None)

        if midi_number is None:
            raise ValueError("Pitch não encontrado nos dados.")
        
        return self.midi_to_note_name(midi_number)
