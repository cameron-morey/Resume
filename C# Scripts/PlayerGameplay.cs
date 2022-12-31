using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class PlayerGameplay : MonoBehaviour, IDamageable
{
    public enum PlayerState
    {
        Aiming,
        Reloading,
        Free
    }

    public PlayerState playerState = PlayerState.Free;

    [SerializeField]
    public float grabDistance;

    [SerializeField]
    private Character_Controller characterController;

    [SerializeField]
    public Gun currentGun;

    [SerializeField]
    private float maxHealth;

    [SerializeField]
    private float currentHealth;

    [Header("States")]
    [SerializeField]
    public bool isAiming = false;

    [SerializeField]
    public bool isReloading;

    [Header("UI")]
    [SerializeField]
    private TextMeshProUGUI pickUpText;

    [SerializeField]
    private Image healthBar;

    //[SerializeField]
    //private List<Magazine> magazines = new List<Magazine>();

    [SerializeField]
    private Camera playerCamera;

    public Inventory playerInventory;

    public BaseGun testBaseGun;

    // Start is called before the first frame update
    void Start()
    {
        currentHealth = maxHealth;
        UpdateHealthBar();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void TakeDamage(float damageTaken)
    {
        currentHealth -= damageTaken;
        UpdateHealthBar();
        Debug.Log("Took Damage");
    }

    public void TakeDamage(float damageTaken, RaycastHit hitdata)
    {
        Debug.Log("Nothing");
    }

    private void UpdateHealthBar()
    {
        healthBar.fillAmount = currentHealth / maxHealth;
    }

    public void Interact()
    {
        CheckInteract();
    }

    public void StopInteract()
    {

    }

    public void Shoot(bool isShooting)
    {
        if (isShooting)
            StartCoroutine(currentGun.Shooting());

        else
            StopCoroutine(currentGun.Shooting());


        currentGun.isShooting = isShooting;

    }

    public void Aim()
    {
        if (!isAiming)
            isAiming = true;

        else if (isAiming)
            isAiming = false;

        currentGun.OnAim();

    }

    public IEnumerator StartReloading()
    {
        if (!isReloading)
        {
            isReloading = true;

            yield return StartCoroutine(currentGun.Reload());

            isReloading = false;
        }
    }

    public void StopReloading()
    {
        if (isReloading)
        {
            isReloading = false;

            StopCoroutine(currentGun.Reload());
        }
    }

    //public void SwapMags(Magazine lastMag)
    //{
    //    magazines.RemoveAt(0);
    //    magazines.Add(lastMag);
    //}

    private void CheckInteract()
    {
        RaycastHit hit;
        Ray ray = playerCamera.ViewportPointToRay(Vector3.one * 0.5f);

        if (Physics.Raycast(ray, out hit, grabDistance))
        {
            if (hit.collider.gameObject.GetComponentInParent<ILootable>() != null)
            {
                hit.collider.gameObject.GetComponentInParent<ILootable>().Loot();

                playerInventory.AddToInventory(GameObject.Find(hit.collider.gameObject.GetComponentInParent<ILootable>().GetParent()));
            }
        }
    }

    public void ToggleInventory()
    {
        playerInventory.ToggleInventory();
    }

    public void WeaponInventory()
    {
        testBaseGun.ToggleInventory();
    }
}
