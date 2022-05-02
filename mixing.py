from pulse import *

pulse1 = Pulse(𝛕=100e-15, 𝜆ₒ=800e-9)
pulse2 = Pulse(𝛕=100e-15, 𝜆ₒ=1040e-9)


for i in range(10):
    pulse1.propagate(1e-3)
    pulse2.propagate(1e-3)
    mixingField = pulse1.field.real * pulse2.field.real
    pulse = Pulse(time=pulse1.time, field=mixingField)
    print(pulse1.distancePropagated)
    pulse.setupPlot("Mixing")
    
    pulse.drawEnvelope()
    pulse.drawField()
    
    plt.draw()
    plt.pause(0.001)

    pulse.tearDownPlot()
