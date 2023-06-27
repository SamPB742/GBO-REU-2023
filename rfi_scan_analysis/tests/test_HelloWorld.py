from rfi_scan_analysis import HelloWorld

def test_value():
    greeting = HelloWorld.sayHello()
    assert greeting == "Hello World!"