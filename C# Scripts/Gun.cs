using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class Gun : MonoBehaviour
{
    private enum FireMode
    {
        Burst,
        Semi,
        Full
    }

    [Header("Recoil Variables")]
    [SerializeField]
    private AnimationCurve xCurve;
    [SerializeField]
    private AnimationCurve yCurve;

    public float minRotation;
    public float maxRotation;

    const float MINROTATION = -100f;
    const float MAXROTATION = 100f;

    private float minRotationScaler;
    private float maxRotationScaler;

    [SerializeField]
    private Vector2 horizontalClamp;
    [SerializeField]
    private Vector2 verticalClamp;
    [SerializeField]
    private Vector2 zClamp;

    [Header("Hipfire Variables")]
    [SerializeField]
    [Range(0f, MINROTATION)]
    private float minHipfireHorizontalRecoil;
    [SerializeField]
    [Range(0f, MAXROTATION)]
    private float maxHipfireHorizontalRecoil;

    [SerializeField]
    [Range(0f, MINROTATION)]
    private float minHipfireVerticalRecoil;
    [SerializeField]
    [Range(0f, MAXROTATION)]
    private float maxHipfireVerticalRecoil;

    [SerializeField]
    [Range(0f, MINROTATION)]
    private float minRecoilHipfireZ;

    [SerializeField]
    [Range(0f, MAXROTATION)]
    private float maxRecoilHipfireZ;

    [Header("ADS Variables")]
    // Does nothing for gun recoil only for camera shaking
    [SerializeField]
    [Range(0f, MINROTATION)]
    private float minADSHorizontalRecoil;
    [SerializeField]
    [Range(0f, MAXROTATION)]
    private float maxADSHorizontalRecoil;

    [SerializeField]
    [Range(0f, MINROTATION)]
    private float minADSVerticalRecoil;
    [SerializeField]
    [Range(0f, MAXROTATION)]
    private float maxADSVerticalRecoil;

    [SerializeField]
    [Range(0f, MINROTATION)]
    private float minRecoilADSZ;

    [SerializeField]
    [Range(0f, MAXROTATION)]
    private float maxRecoilADSZ;

    [Header("Extra Recoil Variable")]
    [SerializeField]
    private float bulletRecoilInit;

    [SerializeField]
    private int bulletRecoilInitCount;

    [SerializeField]
    private float snappiness;
    [SerializeField]
    private float returnSpeed;

    private Vector3 currentRotation;
    private Vector3 targetRotation;

    private List<string> variableNames = new List<string>() { "minHipfireHorizontal", "maxHipfireHorizontal",
    "minHipfireVertical", "maxHipfireVertical", "minHipfireZ", "maxHipfireZ",
    "minADSHorizontal", "maxADSHorizontal", "minADSVertical", "maxADSVertical", "minADSZ", "maxADSZ"};

    [Header("Gun Properties")]
    [SerializeField]
    private FireMode fireMode;

    [SerializeField]
    private float bulletsPerShot;

    [SerializeField]
    private float reloadSpeed;

    //Rounds per Minute
    [SerializeField]
    private float fireRate;

    [SerializeField]
    public int maxAmmo;

    [SerializeField]
    public int maxMag;

    [SerializeField]
    private float scopedFOV;

    [SerializeField]
    private float scopeSpeed;

    [SerializeField]
    private float damageTemp;

    public Magazine magazine;

    [Header("UI Variables")]
    [SerializeField]
    private TextMeshProUGUI currentAmmoText;

    [Header("Gun Customizable Variables")]
    [SerializeField]
    private bool isGunCustomizable;

    [SerializeField]
    private CustomizableGun customizableGunScript;

    [Header("Gun world variables")]
    [SerializeField]
    private GameObject bulletGameObject;

    [SerializeField]
    private Transform bulletSpawnPoint;

    [SerializeField]
    private Camera playerCamera;

    [SerializeField]
    private GameObject gun;

    public BaseGun baseGun;

    [SerializeField]
    private GameObject impactEffect;

    [SerializeField]
    private GameObject headControls;

    [SerializeField]
    private GameObject weaponHolder;

    [SerializeField]
    private Animator animationControls;

    [SerializeField]
    private Transform barrelEnd;

    [SerializeField]
    private Transform bulletEjection;

    [SerializeField]
    private GameObject ejectObject;

    [SerializeField]
    private ParticleSystem barrelSmoke;

    [SerializeField]
    private PlayerGameplay playerScript;

    [SerializeField]
    private float ejectLifeSpan;

    [Header("Script Helper Variables")]

    [SerializeField]
    private bool hasShot = false;

    [SerializeField]
    private bool canShoot = true;

    private bool isReloading = false;

    public bool isShooting = false;

    private bool isScoped = false;

    [SerializeField]
    public int currentMag;

    [SerializeField]
    public int currentAmmo;

    private float normalFOV;

    // Start is called before the first frame update
    void Start()
    {
        currentAmmo = maxAmmo;
        currentMag = magazine.bullets.Count;

        currentAmmoText.text = currentMag.ToString();

        normalFOV = playerCamera.fieldOfView;

        minRotationScaler = MINROTATION / minRotation;
        maxRotationScaler = MAXROTATION / maxRotation;

        if (isGunCustomizable && customizableGunScript != null)
            SetWeaponRecoilValues();

        UpdateRecoilVariables();
    }

    // Update is called once per frame
    void Update()
    {
        targetRotation = Vector3.Lerp(targetRotation, Vector3.zero, returnSpeed * Time.deltaTime);
        currentRotation = Vector3.Slerp(currentRotation, targetRotation, snappiness * Time.fixedDeltaTime);
        //weaponHolder.transform.localRotation = Quaternion.Euler(currentRotation);
        playerCamera.transform.localRotation = Quaternion.Euler(currentRotation);
    }

    private void SetWeaponRecoilValues()
    {
        Dictionary<string, float> values = customizableGunScript.GetRecoilValues();

        minHipfireHorizontalRecoil += values[variableNames[0]];
        maxHipfireHorizontalRecoil -= values[variableNames[1]];

        minHipfireVerticalRecoil += values[variableNames[2]];
        maxHipfireVerticalRecoil -= values[variableNames[3]];

        minRecoilHipfireZ += values[variableNames[4]];
        maxRecoilHipfireZ -= values[variableNames[5]];


        minADSHorizontalRecoil += values[variableNames[6]];
        maxADSHorizontalRecoil -= values[variableNames[7]];

        minADSVerticalRecoil += values[variableNames[8]];
        maxADSVerticalRecoil -= values[variableNames[9]];

        minRecoilADSZ += values[variableNames[10]];
        maxRecoilADSZ -= values[variableNames[11]];
    }


    private void UpdateRecoilVariables()
    {
        minHipfireHorizontalRecoil /= minRotationScaler;
        maxHipfireHorizontalRecoil /= maxRotationScaler;

        minHipfireVerticalRecoil /= minRotationScaler;
        maxHipfireVerticalRecoil /= maxRotationScaler;

        minRecoilHipfireZ /= minRotationScaler;

        maxRecoilHipfireZ /= maxRotationScaler; ;

        minADSHorizontalRecoil /= minRotationScaler;
        maxADSHorizontalRecoil /= maxRotationScaler;

        minADSVerticalRecoil /= minRotationScaler;
        maxADSVerticalRecoil /= maxRotationScaler;

        minRecoilADSZ /= minRotationScaler;

        maxRecoilADSZ /= maxRotationScaler;
    }

    public IEnumerator Reload()
    {
        isReloading = true;
        yield return new WaitForSeconds(reloadSpeed);
        isReloading = false;

        if (currentAmmo >= maxMag)
        {
            currentAmmo -= maxMag;
            currentMag = maxMag;
        }

        if (currentAmmo < maxMag)
        {
            currentMag = currentAmmo;
            currentAmmo = 0;
        }

        //playerScript.SwapMags(magazine);

        //currentMag = nextMag.bullets.Count;
        //magazine = nextMag;

        currentAmmoText.text = currentMag.ToString();
    }

    private void Recoil()
    {
        //Normalize? 45 max/min
        // Vector3(Vertical, Horizontal, Shake for camera no affect on guns)
        if (!playerScript.isAiming)
        {
            float randomVert = -Random.Range(minHipfireVerticalRecoil, maxHipfireVerticalRecoil);
            float randomHort = Random.Range(minHipfireHorizontalRecoil, maxHipfireHorizontalRecoil);
            float randomZ = Random.Range(minRecoilHipfireZ, maxRecoilHipfireZ);

            //Debug.Log(randomVert);
            //Debug.Log(targetRotation);
            //Debug.Log(targetRotation);
            targetRotation += new Vector3(randomVert, randomHort, randomZ);

            targetRotation = new Vector3(Mathf.Clamp(targetRotation.x, verticalClamp.x, verticalClamp.y),
                Mathf.Clamp(targetRotation.y, horizontalClamp.x, horizontalClamp.y),
                Mathf.Clamp(targetRotation.z, zClamp.x, zClamp.y));

            //Debug.Log(targetRotation);
            currentRotation = Vector3.Slerp(currentRotation, targetRotation, snappiness * Time.fixedDeltaTime);
        }

        if (playerScript.isAiming)
        {
            float randomVert = -Random.Range(minADSVerticalRecoil, maxADSVerticalRecoil);
            float randomHort = Random.Range(minADSHorizontalRecoil, maxADSHorizontalRecoil);
            float randomZ = Random.Range(minRecoilADSZ, maxRecoilADSZ);

            targetRotation += new Vector3(randomVert, randomHort, randomZ);

            targetRotation = new Vector3(Mathf.Clamp(targetRotation.x, verticalClamp.x, verticalClamp.y),
                Mathf.Clamp(targetRotation.y, horizontalClamp.x, horizontalClamp.y),
                Mathf.Clamp(targetRotation.z, zClamp.x, zClamp.y));

            currentRotation = Vector3.Slerp(currentRotation, targetRotation, snappiness * Time.fixedDeltaTime);
        }
    }

    private bool CanShoot()
    {
        if (isReloading)
            return false;

        if (currentMag == 0)
            return false;

        if (hasShot && fireMode != FireMode.Full)
            return false;

        return true;
    }

    private IEnumerator TimeTillNextShot()
    {
        canShoot = false;
        yield return new WaitForSeconds(60f / fireRate);

        canShoot = true;
        yield return null;
    }

    public IEnumerator Shooting()
    {
        if (canShoot)
        {
            do
            {
                if (fireMode == FireMode.Burst)
                {
                    for (int i = 0; i < bulletsPerShot; i++)
                    {
                        Shoot();
                        yield return StartCoroutine(TimeTillNextShot());
                    }

                    hasShot = true;
                }
                else
                {
                    Shoot();
                    hasShot = true;
                    yield return StartCoroutine(TimeTillNextShot());
                }

            } while (isShooting && CanShoot());

            hasShot = false;
        }
    }

    public IEnumerator CreateEjectObject()
    {
        GameObject newEject = Instantiate(ejectObject, bulletEjection.position, Quaternion.Euler(ejectObject.transform.right));
        newEject.GetComponent<Rigidbody>().AddRelativeForce((ejectObject.transform.right + ejectObject.transform.up), ForceMode.Impulse);

        yield return new WaitForSeconds(ejectLifeSpan);

        Destroy(newEject);
    }

    public void Shoot()
    {
        if (CanShoot())
        {
            //float damage = magazine.bullets[0].damage;
            //float armorPen = magazine.bullets[0].armorPen;
            //Bullet.BulletType bulletType = magazine.bullets[0].bulletType;

            //Debug.Log($"Bullet Type {bulletType}, damage {damage}, armor Pen {armorPen}");
            //magazine.Fired();

            StartCoroutine(CreateEjectObject());
            barrelSmoke.Play();

            RaycastHit hit;

            if (Physics.Raycast(bulletSpawnPoint.position, bulletSpawnPoint.forward, out hit, float.MaxValue))
            {
                StartCoroutine(ParticleLife(Instantiate(impactEffect, hit.point, Quaternion.LookRotation(hit.normal))));

                if (hit.collider.tag == "Damageable")
                {
                    hit.collider.GetComponentInParent<IDamageable>().TakeDamage(damageTemp, hit);
                }
            }

            Recoil();
            currentMag--;
            currentAmmoText.text = currentMag.ToString();
        }
    }

    public void OnAim()
    {
        isScoped = !isScoped;
        animationControls.SetBool("isScoped", isScoped);

        if (isScoped)
        {
            StartCoroutine(OnAimStart());
        }

        else
        {
            StartCoroutine(OnAimEnd());
        }
    }

    private IEnumerator OnAimStart()
    {
        yield return new WaitForSeconds(scopeSpeed);

        playerCamera.fieldOfView = scopedFOV;

    }

    private IEnumerator OnAimEnd()
    {
        playerCamera.fieldOfView = normalFOV;

        yield return new WaitForSeconds(scopeSpeed);
    }

    private IEnumerator ParticleLife(GameObject effect)
    {
        yield return new WaitForSeconds(10f);
        Destroy(effect);
    }
}