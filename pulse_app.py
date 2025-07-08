from mytk import *


class PulseApp(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.window.title = "Pulse properties calculator"
        self.properties_box = Box(
            label="Pulse properties", width=400, height=200
        )
        self.properties_box.grid_into(
            self.window, column=1, row=0, pady=10, padx=10, sticky="nsew"
        )
        self.properties_box.widget.grid_propagate(False)

        Label("Pulse width").grid_into(
            self.properties_box, row=0, column=0, pady=5, padx=10, sticky="nse"
        )
        self.pulse_width = FormattedEntry(character_width=6)
        self.pulse_width.grid_into(
            self.properties_box, row=0, column=1, pady=5, padx=10, sticky="nse"
        )
        self.pulse_width.value_variable.set(100)

        self.pulse_width_units = PopupMenu(["fs", "ps"])
        self.pulse_width_units.value_variable = StringVar(value="fs")
        self.pulse_width_units.grid_into(
            self.properties_box, row=0, column=2, pady=5, padx=10, sticky="w"
        )

        Label("Spectral width").grid_into(
            self.properties_box, row=1, column=0, pady=5, padx=10, sticky="nse"
        )
        self.spectral_width = FormattedEntry(character_width=6)
        self.spectral_width.grid_into(
            self.properties_box, row=1, column=1, pady=5, padx=10, sticky="nse"
        )
        self.spectral_width.value_variable.set(100)

        self.spectral_width_units = PopupMenu(["nm", "THz"])
        self.spectral_width_units.value_variable = StringVar(value="nm")
        self.spectral_width_units.grid_into(
            self.properties_box, row=1, column=2, pady=5, padx=10, sticky="w"
        )

        Label("Central wavelength").grid_into(
            self.properties_box, row=2, column=0, pady=5, padx=10, sticky="nse"
        )
        self.wavelength = FormattedEntry(character_width=6)
        self.wavelength.grid_into(
            self.properties_box, row=2, column=1, pady=5, padx=10, sticky="nse"
        )
        self.wavelength.value_variable.set(800)

        self.wavelength_units = PopupMenu(["µm", "nm"])
        self.wavelength_units.value_variable = StringVar(value="nm")
        self.wavelength_units.grid_into(
            self.properties_box, row=2, column=2, pady=5, padx=10, sticky="w"
        )


if __name__ == "__main__":
    app = PulseApp()
    app.mainloop()
