// eSort_main_final_auto_clear_v2_with_buzzer.ino
// Added active buzzer that beeps in sync with hazard red LED

#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// --- Pins ---
const int PIN_BASE = 9;
const int PIN_TILT = 10;
const int PIN_HAZARD_LED = 7;  // red LED for hazardous indicator (blinking)
const int PIN_MOTOR_LED = 6;   // yellow LED to indicate motors working
const int PIN_IDLE_LED  = 8;   // green LED to indicate idle/ready
const int PIN_BUZZER    = 5;   // active buzzer (will beep with hazard blink)

// --- LCD I2C settings ---
const uint8_t LCD_ADDR = 0x27; // change to 0x3F if needed
const int LCD_COLS = 16;
const int LCD_ROWS = 2;
LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);

// --- Calibrated constants (example values) ---
const int BASE_NEUTRAL          = 90;
const int BASE_POS_BIN_LEFT     = 45;
const int BASE_POS_BIN_RIGHT    = 135;
const int TILT_NEUTRAL          = 90;
const int TILT_DROP_FORWARD     = 130;
const int TILT_DROP_BACKWARD    = 50;
// ----------------------------------------------------------------------

const int STEP_DEG      = 1;       // degrees per step for smooth motion
const int STEP_DELAY_MS = 10;      // ms delay between each step
const unsigned long HOLD_MS = 5000;     // how long to hold tilt at drop (ms)
const unsigned long SETTLE_MS = 150;    // pause after each move

// Hazard blink interval
const unsigned long HAZARD_BLINK_MS = 500UL;     // LED blink & buzzer interval

Servo baseServo;
Servo tiltServo;

// Hazard state
bool hazardActive = false;
unsigned long lastBlink = 0;
bool ledState = false;

// helper led/buzzer state setters
inline void setIdleLED(bool on)   { digitalWrite(PIN_IDLE_LED, on ? HIGH : LOW); }
inline void setMotorLED(bool on)  { digitalWrite(PIN_MOTOR_LED, on ? HIGH : LOW); }
inline void setHazardLED(bool on) { digitalWrite(PIN_HAZARD_LED, on ? HIGH : LOW); }
inline void setBuzzer(bool on)    { digitalWrite(PIN_BUZZER, on ? HIGH : LOW); }

int clampAngle(int v){ if(v<0) return 0; if(v>180) return 180; return v; }

void smoothMove(Servo &s, int fromDeg, int toDeg){
  // NOTE: blocking (uses delay). Motor LED is turned on/off by caller.
  fromDeg = clampAngle(fromDeg);
  toDeg   = clampAngle(toDeg);
  if (fromDeg == toDeg) { s.write(toDeg); delay(SETTLE_MS); return; }
  if (toDeg > fromDeg) {
    for (int a = fromDeg; a <= toDeg; a += STEP_DEG){ s.write(a); delay(STEP_DELAY_MS); }
  } else {
    for (int a = fromDeg; a >= toDeg; a -= STEP_DEG){ s.write(a); delay(STEP_DELAY_MS); }
  }
  s.write(toDeg);
  delay(SETTLE_MS);
}

// helper: center & trim text to a 12-char area then print on row (0 or 1)
void lcdPrintCentered12(const String &text, int row) {
  int maxLen = 12;
  String s = text;
  if (s.length() > maxLen) s = s.substring(0, maxLen);
  int startIn12 = (maxLen - s.length()) / 2;
  int col = 2 + startIn12;
  lcd.setCursor(col, row);
  lcd.setCursor(2, row);
  for (int i=0;i<12;i++) lcd.print(' ');
  lcd.setCursor(col, row);
  lcd.print(s);
}

void lcdShowTwoCentered(const String &l1, const String &l2) {
  lcd.clear();
  lcdPrintCentered12(l1, 0);
  lcdPrintCentered12(l2, 1);
}

void performSmoothBinAction(int bin) {
  if (bin < 1 || bin > 5) return;
  if (bin == 5) return; // hazardous handled separately

  int baseTarget = BASE_NEUTRAL;
  if (bin == 1 || bin == 4) baseTarget = BASE_POS_BIN_LEFT;
  else if (bin == 2 || bin == 3) baseTarget = BASE_POS_BIN_RIGHT;

  int tiltTarget = TILT_NEUTRAL;
  if (bin == 1 || bin == 2) tiltTarget = TILT_DROP_FORWARD;
  else if (bin == 3 || bin == 4) tiltTarget = TILT_DROP_BACKWARD;

  // indicate motors active: turn off idle LED, turn on motor LED
  setIdleLED(false);
  setMotorLED(true);

  // move base then tilt then back
  smoothMove(baseServo, BASE_NEUTRAL, baseTarget);
  delay(120);
  smoothMove(tiltServo, TILT_NEUTRAL, tiltTarget);
  delay(HOLD_MS);
  smoothMove(tiltServo, tiltTarget, TILT_NEUTRAL);
  smoothMove(baseServo, baseTarget, BASE_NEUTRAL);

  // finished moving: restore motor/idles
  setMotorLED(false);
  setIdleLED(true);
}

// mapping bin number -> display-friendly label (max ~12 chars)
String binLabel(int bin) {
  switch(bin) {
    case 1: return "Biodegradable";  // 12 chars
    case 2: return "Recyclable";     // 10 chars
    case 3: return "Paper";          // 5 chars
    case 4: return "Non-Recycl";     // 9 chars
    case 5: return "HAZARDOUS";      // 8 chars
    default: return "Unknown";
  }
}

void activateHazardRoutine() {
  hazardActive = true;
  lastBlink = millis();
  ledState = true;
  // red hazard blinking starts; ensure motor/idle reflect non-idle
  setHazardLED(true);
  setBuzzer(true);           // start buzzer on initial show
  setMotorLED(false);
  setIdleLED(false);
  // show hazard message (centered top, left-aligned second line)
  lcd.clear();
  lcdPrintCentered12("!! HAZARD !!", 0);
  lcd.setCursor(0, 1);
  lcd.print("Remove waste");
}

void deactivateHazardRoutine() {
  hazardActive = false;
  setHazardLED(false);
  setBuzzer(false);
  lcd.clear();
  lcdPrintCentered12("eSort ready", 0);
  delay(600);
  lcd.clear();
  lcdPrintCentered12("Place waste", 0);
  // restore idle LED
  setIdleLED(true);
}

void setup(){
  Serial.begin(115200);
  baseServo.attach(PIN_BASE);
  tiltServo.attach(PIN_TILT);

  // LED & buzzer pin modes
  pinMode(PIN_HAZARD_LED, OUTPUT);
  pinMode(PIN_MOTOR_LED, OUTPUT);
  pinMode(PIN_IDLE_LED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  // initial states
  setHazardLED(false);
  setMotorLED(false);
  setBuzzer(false);
  setIdleLED(true);

  Wire.begin();
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcdPrintCentered12("eSort starting", 0);
  delay(800);
  lcd.clear();
  lcdPrintCentered12("Place waste", 0);

  baseServo.write(BASE_NEUTRAL);
  tiltServo.write(TILT_NEUTRAL);
  delay(500);
  Serial.println("ARDUINO READY");
}

String inLine = "";

void loop(){
  // handle hazard blinking (non-blocking)
  if (hazardActive) {
    unsigned long now = millis();
    if (now - lastBlink >= HAZARD_BLINK_MS) {
      ledState = !ledState;
      setHazardLED(ledState);   // blink red LED
      setBuzzer(ledState);      // beep in sync with red LED
      lastBlink = now;
    }
    // continue to parse serial while blinking
  }

  // Serial parsing (non-blocking)
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') { /* ignore */ }
    else if (c == '\n') {
      inLine.trim();
      if (inLine.length() > 0) {
        Serial.println("RECEIVED"); // immediate ack
        if (inLine.startsWith("C:")) {
          int bin = inLine.substring(2).toInt();
          if (bin >= 1 && bin <= 5) {
            if (bin == 5) {
              // hazardous: go into hazard mode (wait until CLEAR)
              activateHazardRoutine();
              // DO NOT print DONE here — wait for CLEAR from Python
            } else {
              // show "Moving to.." then compartment label
              String label = binLabel(bin);
              lcdShowTwoCentered("Moving to..", label);

              // perform movement (motor LED handled inside)
              performSmoothBinAction(bin);

              // show Placed
              String placed = "Placed in " + String(bin);
              lcdShowTwoCentered(placed, "");
              Serial.println("DONE");
              delay(600);
              // return to idle
              lcdShowTwoCentered("eSort ready", "Place waste");
              // ensure idle LED ON
              setIdleLED(true);
            }
          } else {
            Serial.println("ERR:BAD_BIN");
          }
        } else if (inLine == "CLEAR") {
          // external clear from Python: stop hazard and return DONE
          if (hazardActive) {
            deactivateHazardRoutine();
            Serial.println("DONE");
          } else {
            Serial.println("DONE");
          }
        } else {
          Serial.println("ERR:UNKNOWN_CMD");
        }
      }
      inLine = "";
    } else {
      inLine += c;
    }
  }
}
